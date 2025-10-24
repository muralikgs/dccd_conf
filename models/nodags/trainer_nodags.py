import torch 
import warnings 

import numpy as np 

from torch.utils.data import DataLoader
from sklearn.covariance import graphical_lasso 

from data_generation.simulation_data import InterventionDataset

from models.nodags.resblock import iResBlock
from models.helper.layers.mlpLipschitz import linearLipschitz

def update_lipschitz(model, n_iterations):
    for m in model.modules():
        if isinstance(m, linearLipschitz):
            m.compute_weight(update=True, n_iterations=n_iterations)

class NODAGSTrainer: 

    def __init__(
            self,
            nodags: iResBlock,
            obs=False, 
            max_epochs=500, 
            batch_size=1024, 
            lr=1e-2, 
            lambda_c=1e-2, 
            lambda_dag=10,
            rho=1e-2, 
            n_lip_iterations=5,
            glasso_iters=2
            ):
        
        # model and parameters
        self.model = nodags
        self.obs = obs 

        # training parameters
        self.max_epochs = max_epochs 
        self.batch_size = batch_size 
        self.lr = lr
        self.lambda_c = lambda_c 
        self.lambda_dag = lambda_dag
        self.n_lip_iterations = n_lip_iterations

        # graphical lasso parameters
        self.rho = rho 
        self.glasso_iters = glasso_iters

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def train(
            self,
            intervention_datasets, 
            intervention_targets, 
            print_interval=100,
            verbose=False,
    ):
        
        intervention_dataloaders = list()
        for dataset, targets in zip(intervention_datasets, intervention_targets):
            intervention_dataloaders.append(
                DataLoader(
                    dataset=InterventionDataset(dataset, targets),
                    batch_size=self.batch_size
                    )
            )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        count = 0

        self.model = self.model.to(device) 

        self.loss_list = list() 
        self.iteration_list = list() 

        self.model.init_beta_mat(intervention_targets)
        for epoch in range(self.max_epochs):
            for inter, inter_dataloader in enumerate(intervention_dataloaders):
                
                self.model.generate_current_distribution_params(intervention_targets=intervention_targets[inter])

                for it, batch in enumerate(inter_dataloader):
                    count += 1
                    self.optimizer.zero_grad()

                    # Step - 1 (Update NN and graph parameters)

                    X, masks = batch[0].float().to(device), batch[1].float().to(device)

                    e, loss_pen, loss, _, h_w = self.model.logprob(
                        X, 
                        masks, 
                        lambda_c=self.lambda_c,
                        lambda_dag=self.lambda_dag, 
                        obs=self.obs
                    )

                    loss_pen.backward()
                    self.optimizer.step()
                    update_lipschitz(self.model, n_iterations=self.n_lip_iterations)

                    if self.model.confounders:

                        # Step - 2 (Update confounder + noise parameters)

                        with torch.no_grad():
                            observational_mask = masks == 1
                            n_observed = observational_mask[0].sum()
                            marginal_e = e[observational_mask].view(e.shape[0], n_observed)
                            S = np.cov(marginal_e.detach().cpu().numpy().T)
                            # print(np.abs(S).max(), np.abs(S).min())
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")  # Ignore all warnings
                                cov, prec = graphical_lasso(S, alpha=self.rho, max_iter=self.glasso_iters)

                            self.model.inter_cov_mat = torch.tensor(cov, dtype=torch.float, device=device)
                            self.model.inter_prec_mat = torch.tensor(prec, dtype=torch.float, device=device)

                    if verbose and (count % print_interval == 0):
                        dag_const = h_w if self.obs else -1
                        print("Epoch: {}/{}, Intervention: {}/{}, Iteration: {}/{}, Loss: {:.2f}, h(W): {:.2f}".format(
                            epoch+1, 
                            self.max_epochs,
                            inter+1, 
                            len(intervention_targets),
                            it+1, 
                            len(inter_dataloader),
                            loss.item(),
                            dag_const), end="\r", flush=True
                            )
                        
                        self.loss_list.append(loss.item())
                        self.iteration_list.append(count)

                self.model.update_distribution_params(intervention_targets=intervention_targets[inter])

        print()


        
        