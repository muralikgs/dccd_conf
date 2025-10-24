import torch 
import warnings 
import time

import numpy as np 

from torch.utils.data import DataLoader
from sklearn.covariance import graphical_lasso 

from data_generation.simulation_data import InterventionDataset

from models.dccd.implicitblock import imBlock
from models.helper.layers.mlpLipschitz import linearLipschitz

def update_lipschitz(model, n_iterations):
    for m in model.modules():
        if isinstance(m, linearLipschitz):
            m.compute_weight(update=True, n_iterations=n_iterations)

class DCCDTrainer: 

    def __init__(
            self, 
            model: imBlock,
            max_epochs=500,
            batch_size=512, 
            lr=1e-2, 
            lambda_c=1e-2, 
            rho=1e-2, 
            n_lip_iterations=5,
            glasso_iters=2
    ):
        
        # model and parameters
        self.model = model 

        # training parameters
        self.max_epochs = max_epochs 
        self.batch_size = batch_size
        self.lr = lr 
        self.lambda_c = lambda_c 
        self.n_lip_iterations = n_lip_iterations 

        # graph lasso parameters 
        self.rho = rho 
        self.glasso_iters = glasso_iters 

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr) 

    def train(
            self,
            intervention_datasets, 
            intervention_targets, 
            verbose=False, 
            sleep_time=False
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

        self.model = self.model.to(device) 

        self.loss_list = list() 
        self.logpx_list = list() 

        for epoch in range(self.max_epochs): 
            av_log = 0
            for inter, inter_dataloader in enumerate(intervention_dataloaders):

                self.model.generate_current_distribution_params(intervention_targets=intervention_targets[inter])

                for it, batch in enumerate(inter_dataloader): 
                    self.optimizer.zero_grad() 

                    x, mask = batch[0].float().to(device), batch[1].float().to(device)
                    z, logdetgrad, logpx, loss_w_pen = self.model.loss(x, mask, lambda_c=self.lambda_c)
                    av_log += logpx.item()
                    loss_w_pen.backward() 
                    self.optimizer.step() 
                    update_lipschitz(self.model, n_iterations=self.n_lip_iterations)

                    if self.model.confounders: 
                        with torch.no_grad(): 
                            observational_mask = mask == 1
                            n_observed = observational_mask[0].sum() 
                            marginal_z = z[observational_mask].view(z.shape[0], n_observed)
                            S = np.cov(marginal_z.detach().cpu().numpy().T)
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore") # Ignore all warnings
                                cov, prec = graphical_lasso(S, alpha=self.rho, max_iter=self.glasso_iters)

                            self.model.inter_cov_mat = torch.tensor(cov, dtype=torch.float, device=device)
                            self.model.inter_prec_mat = torch.tensor(prec, dtype=torch.float, device=device)
                
                self.model.update_distribution_params(intervention_targets=intervention_targets[inter])
                av_log /= len(inter_dataloader)
                if verbose:
                    if sleep_time: 
                        time.sleep(1)
                    print(f"Epoch: {epoch+1}/{self.max_epochs}, Intervention: {inter+1}/{len(intervention_targets)}, log P(X): {av_log:.2f}", end="\r", flush=True)        
                self.loss_list.append(loss_w_pen.item())
                self.logpx_list.append(av_log)
                av_log = 0

        print()



