
import torch
import torch.nn as nn 
import numpy as np 
import math 
import time 
import os

from models.helper.functions import gumbelSoftMLP

def standard_normal_logprob_indep(z, noise_scales):
    logZ = -0.5 * torch.log(2 * math.pi * (noise_scales**(2)))#(noise_scales.pow(2))) # change this back to pow
    return logZ - z.pow(2) / (2 * (noise_scales**(2)))

def standard_normal_logprob(z, noise_precision_mat):
    logZ = 0.5 * torch.log(torch.abs(torch.linalg.det(noise_precision_mat))) - (z.shape[1]/2) * torch.log(2 * torch.tensor(torch.pi))
    # print(logZ)
    return logZ - (z @ noise_precision_mat @ z.T).diag().view(-1,1) / 2

def necessary_normal_logprob(z, noise_precision_mat):
    return - (z @ noise_precision_mat @ z.T).diag().view(-1, 1) / 2

def dag_constraint(W, s=1, method='log-det'):
    if method == 'expm':
        return torch.trace(torch.matrix_exp(W * W)) - W.shape[0]
    elif method == 'log-det':
        return -torch.log(s * torch.det(torch.eye(W.shape[0], device=W.device) - W * W)) + W.shape[0] * math.log(s)

def soft_threshold(x, t):
    return torch.sign(x) * ((x.abs() - t) > 0) * (x.abs() - t)

def single_step_cgd(V, u, beta, rho=0.2, n_iter=50):

    # p = V.shape[0]
    beta_list = list()
    diag_mask = torch.ones_like(V)
    diag_mask.fill_diagonal_(0)
    for _ in range(n_iter):
        beta = soft_threshold(
            x = u - (V * diag_mask) @ beta,
            t = rho
        ) / V.diag().view(-1,1)   

        beta_list.append(beta.squeeze())

    return beta.squeeze(), beta_list

class iResBlock(nn.Module):
    """
    ----------------------------------------------------------------------------------------
    The class for a single residual map, i.e., (id -f)(x) = e. 
    ----------------------------------------------------------------------------------------
    The forward method computes the residual map and also log-det-Jacobian of the map. 

    Parameters:
    1) func - (nn.Module) - torch module for modelling the function f in (I - f).
    2) n_power_series - (int/None) - Number of terms used for computing determinent of log-det-Jac, 
                                     set it to None to use Russian roulette estimator. 
    3) neumann_grad - (bool) - If True, Neumann gradient estimator is used for Jacobian.
    4) n_dist - (string) - distribution used to sample n when using Russian roulette estimator. 
                           'geometric' - geometric distribution.
                           'poisson' - poisson distribution.
    5) lamb - (float) - parameter of poisson distribution.
    6) geom_p - (float) - parameter of geometric distribution.
    7) n_samples - (int) - number of samples to be sampled from n_dist. 
    8) grad_in_forward - (bool) - If True, it will store the gradients of Jacobian with respect to 
                                  parameters in the forward pass. 
    9) n_exact_terms - (int) - Minimum number of terms in the power series. 
    """
    def __init__(self, 
                 func : gumbelSoftMLP,
                 n_power_series, 
                 neumann_grad=True, 
                 n_dist='geometric', 
                 lamb=2., 
                 geom_p=0.5, 
                 n_samples=1, 
                 grad_in_forward=False, 
                 n_exact_terms=2, 
                 lin_logdet=False, 
                 centered=True, 
                 cov_given=False, 
                 cov=None, 
                 learn_full_cov=False,
                 confounders=True):
        
        super(iResBlock, self).__init__()
        self.f = func
        self.geom_p = nn.Parameter(torch.tensor(np.log(geom_p) - np.log(1. - geom_p)))
        self.lamb = nn.Parameter(torch.tensor(lamb))
        self.n_dist = n_dist
        self.n_power_series = n_power_series 
        self.neumann_grad = neumann_grad 
        self.grad_in_forward = grad_in_forward
        self.n_exact_terms = n_exact_terms
        self.n_samples = n_samples
        self.lin_logdet = lin_logdet
        self.centered = centered
        self.confounders = confounders

        if not confounders:
            self.var = nn.Parameter(torch.ones(self.f.n_nodes, dtype=torch.float))

        # these parameters are updated in the second phase of optimization
        self.cov_mat = torch.eye(self.f.n_nodes)
        self.prec_mat = torch.eye(self.f.n_nodes)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cov_mat = self.cov_mat.to(device)
        self.prec_mat = self.prec_mat.to(device)

        if not learn_full_cov: 
            if cov_given:
                self.cov_mat = torch.tensor(cov)
                self.prec_mat = torch.linalg.inv(self.cov_mat)
        else:
            self.prec_mat = nn.Parameter(self.prec_mat)
        
        if not self.centered:
            self.mu = nn.Parameter(torch.zeros(self.f.n_nodes))
        else:
            self.mu = torch.zeros(self.f.n_nodes) .to(device)

        self.Lambda = nn.Parameter(torch.zeros(self.f.n_nodes))
        # self.Lambda = torch.zeros(self.f.n_nodes).to(device)

    # Function that initializes the beta matrices used in updating the covariance
    # matrix in the graphical lasso step 
    def init_beta_mat(self, intervention_sets):
        self.beta_mats = list()
        for targets in intervention_sets:
            if targets[0] != None:
                n_observations = self.f.n_nodes - len(targets)
            else:
                n_observations = self.f.n_nodes

            self.beta_mats.append(
                torch.zeros(n_observations-1, n_observations)
            )

    # Function that computes the covariance and the precision matrix of the noise 
    # distribution at the start of processing one interventional setting
    def generate_current_distribution_params(self, intervention_targets):
        observed_node_index = np.setdiff1d(np.arange(self.f.n_nodes), intervention_targets)
        self.inter_cov_mat = self.cov_mat[observed_node_index, :][:, observed_node_index]
        self.inter_prec_mat = torch.inverse(self.inter_cov_mat)

    # Function that updates the covariance and the precision matrix of the noise 
    # distribution at the end of processing one interventional setting
    def update_distribution_params(self, intervention_targets):
        observed_node_index = np.setdiff1d(np.arange(self.f.n_nodes), intervention_targets)
        cov_mat_temp = self.cov_mat[observed_node_index, :]
        cov_mat_temp[:, observed_node_index] = self.inter_cov_mat 
        self.cov_mat[observed_node_index, :] = cov_mat_temp 
        self.prec_mat = torch.inverse(self.cov_mat)

    # Function that maps the input data x to latent variables e (confounder + intrinsic noise)
    def forward(self, x, mask, logdet=False, neumann_grad=True):

        self.neumann_grad = neumann_grad

        # Precondition the input to the causal mechanism F 
        # This is primarily used for learning non-contractive dags, but also 
        # works for contractive cyclic graphs
        Lamb_mat = torch.diag(torch.exp(self.Lambda))
        Lamb_mat_inv = torch.diag(1/torch.exp(self.Lambda))
        x_inp = (x - self.mu) @ Lamb_mat

        # Return the log determinant of the Jacobian
        if logdet:
            f_x, logdetgrad, _ = self._logdetgrad(x_inp, mask)
            return (x - self.mu) - (f_x @ Lamb_mat_inv) * mask, logdetgrad
        
        else:
            f_x = self.f(x_inp)
            return (x - self.mu) - (f_x @ Lamb_mat_inv) * mask

    # Function that simulates the SEM given the latent varialbes e (confounder + intrinsic noise) 
    def predict_from_latent(self, latent_vec, mask, x_init=None, n_iter=20, threshold=1e-4):
        x = torch.randn(latent_vec.size(), device=latent_vec.device)
        mask_cmp = torch.ones_like(mask) - mask
        c = x_init * mask_cmp

        if self.dag_input:
            Lamb_mat = torch.diag(torch.exp(self.Lambda))
            Lamb_mat_inv = torch.diag(1/torch.exp(self.Lambda))

        for _ in range(n_iter):
            x_t = x
            x_inp = x - self.mu 
            if self.dag_input:
                x_inp = (x - self.mu) @ Lamb_mat
                f_x = self.f(x_inp) @ Lamb_mat_inv
            else:
                f_x = self.f(x_inp)

            x = f_x * mask + (latent_vec + self.mu) * mask + c 
            if torch.norm(x_t - x) < threshold:
                break
    
        return x 

    # Function that computes the logdeterminant of the Jacobian matrix for the
    # functional map (id - F)
    def _logdetgrad(self, x, mask):
        with torch.enable_grad():
            if self.n_dist == 'geometric':
                geom_p = torch.sigmoid(self.geom_p).item()
                sample_fn = lambda m: geometric_sample(geom_p, m)
                rcdf_fn = lambda k, offset: geometric_1mcdf(geom_p, k, offset)
            elif self.n_dist == 'poisson':
                lamb = self.lamb.item()
                sample_fn = lambda m: poisson_sample(lamb, m)
                rcdf_fn = lambda k, offset: poisson_1mcdf(lamb, k, offset)
            
            # if self.training:
            if self.n_power_series is None:
                # Unbiased estimation.
                lamb = self.lamb.item()
                n_samples = sample_fn(self.n_samples)
                n_power_series = max(n_samples) + self.n_exact_terms
                coeff_fn = lambda k: 1 / rcdf_fn(k, self.n_exact_terms) * \
                    sum(n_samples >= k - self.n_exact_terms) / len(n_samples)
            else:
                # Truncated estimation.
                n_power_series = self.n_power_series
                coeff_fn = lambda k: 1.

            vareps = torch.randn_like(x)

            if self.lin_logdet:
                estimator_fn = linear_logdet_estimator
            else:
                # if self.training and self.neumann_grad:
                if self.neumann_grad:
                    estimator_fn = neumann_logdet_estimator
                else:
                    estimator_fn = basic_logdet_estimator

            if self.training and self.grad_in_forward:
                f_x, logdetgrad = mem_eff_wrapper(
                    estimator_fn, self.f, x, n_power_series, vareps, coeff_fn, self.training
                )
            else:
                x = x.requires_grad_(True)
                f_x = self.f(x)
                tic = time.time()
                if self.lin_logdet:
                    Weight = self.f.layer.weight
                    self_loop_mask = torch.ones_like(Weight)
                    ind = np.diag_indices(Weight.shape[0])
                    self_loop_mask[ind[0], ind[1]] = 0 
                    logdetgrad = estimator_fn(mask * self_loop_mask * Weight, x.shape[0])
                else:
                    logdetgrad = estimator_fn(f_x * mask, x, n_power_series, vareps, coeff_fn, self.training)
                toc = time.time()
                comp_time = toc - tic 

        return f_x, logdetgrad.view(-1, 1), comp_time
    
    # Function that computes the log prob of p(X)
    def logprob(self, 
               x, 
               mask, 
               lambda_c=1e-2, 
               lambda_dag=1, 
               obs=False, 
               fun_type='gst-mlp', 
               s=1, 
               method='expm', 
               neumann_grad=True):

        e, logdetgrad = self.forward(x, mask, logdet=True, neumann_grad=neumann_grad)
        
        lat_prec_mat = self.inter_prec_mat

        # ####TESTING BLOCK####
        # lat_std = torch.exp(self.var)
        # logpe = (standard_normal_logprob(e, noise_scales=lat_std) * mask).sum(1, keepdim=True)
        # ####END OF TESTING BLOCK#### 

        if not self.confounders:
            lat_std = torch.exp(self.var)
            logpe = (standard_normal_logprob_indep(e, noise_scales=lat_std) * mask).sum(1, keepdim=True)

        logpe = standard_normal_logprob(e[:, mask[0]==1], lat_prec_mat).view(-1, 1)
        logpx = logpe + logdetgrad

        loss = -torch.mean(logpx)
        if fun_type == 'fac-mlp' or fun_type == 'gst-mlp':
            l1_norm = self.f.get_w_adj().abs().sum()
        else:
            l1_norm = sum(p.abs().sum() for p in self.parameters())

        loss_pen = loss + lambda_c * l1_norm

        if obs:
            if fun_type == "gst-mlp":
                h_w = dag_constraint(self.f.get_w_adj().abs(), s=s, method=method)
            elif fun_type == "lin-mlp":
                w = self.f.layer.weight.T
                h_w = dag_constraint(w, method=method)

            loss_pen += lambda_dag * h_w
            return e, loss_pen, loss, torch.mean(logdetgrad), h_w

        return e, loss_pen, loss, torch.mean(logpe), torch.mean(logdetgrad)

    # Function to perform graphical Lasso to update the parameters of noise distribution
    # (confounders + intrinsic noise) in each iteration
    def graphical_lasso(self, x, mask, intervention_id, rho=0.2, latent_given=False, latents=None, n_cgd_iters=50):

        n_nodes = self.f.n_nodes

        # get the latent variable
        if latent_given:
            e = latents
        else:
            e = self.forward(x, mask, logdet=False)
        
        # marginalize the latents to only the purely observed variables
        observations_mask = mask == 1
        n_observed = observations_mask[0].sum()
        marginal_e = e[observations_mask].view(e.shape[0], n_observed)
        S = torch.cov(marginal_e.T)

        # loop over each row/column of the covariance matrix
        beta_list = list()
        for i in range(n_observed):
            i_minus_index = np.arange(n_observed) != i 
            W_11 = self.inter_cov_mat[i_minus_index, :][:, i_minus_index]
            s_12 = S[i_minus_index, i].view(-1, 1)

            # solving the lasso problem via coordinate gradient descent
            self.beta_mats[intervention_id][:,i], beta_list_i = single_step_cgd(
                W_11, 
                s_12, 
                self.beta_mats[intervention_id][:,i].view(-1,1), 
                rho,
                n_iter=n_cgd_iters
            ) # CODE TO SOLVE LASSO
            beta_list.append(beta_list_i)

            # print("Beta")
            # print(self.beta_mats)

            # updating the covariance matrix
            w_12 = W_11 @ self.beta_mats[intervention_id][:, i].view(-1,1) 

            self.inter_cov_mat[i_minus_index, i] = w_12.squeeze()
            self.inter_cov_mat[i, i_minus_index] = w_12.squeeze()
            self.inter_cov_mat[i, i] = S[i, i] + rho

            # updating the precision matrix
            self.inter_prec_mat[i, i] = 1 / (self.inter_cov_mat[i, i] - w_12.T @ self.beta_mats[intervention_id][:,i].view(-1,1))
            self.inter_prec_mat[i_minus_index, i] = -self.beta_mats[intervention_id][:,i] * self.inter_prec_mat[i, i]

            # print("Covariance")
            # print(self.inter_cov_mat)

            # print()
        return beta_list


    def get_w_adj(self):
        return self.f.get_w_adj().detach().cpu().numpy()
    
    def save_model(self, path):
        torch.save(self.state_dict(), os.path.join(path, "model.pth"))
        np.save(os.path.join(path, "cov_mat.npy"), self.cov_mat.detach().cpu().numpy())
        np.save(os.path.join(path, "prec_mat.npy"), self.prec_mat.detach().cpu().numpy())

def basic_logdet_estimator(g, x, n_power_series, vareps, coeff_fn, training):
    vjp = vareps
    logdetgrad = torch.tensor(0.).to(x)
    for k in range(1, n_power_series + 1):
        vjp = torch.autograd.grad(g, x, vjp, create_graph=training, retain_graph=True)[0]
        tr = torch.sum(vjp.view(x.shape[0], -1) * vareps.view(x.shape[0], -1), 1)
        delta = -1 / k * coeff_fn(k) * tr
        logdetgrad = logdetgrad + delta
    return logdetgrad

def neumann_logdet_estimator(g, x, n_power_series, vareps, coeff_fn, training):
    vjp = vareps
    neumann_vjp = vareps
    with torch.no_grad():
        for k in range(1, n_power_series + 1):
            vjp = torch.autograd.grad(g, x, vjp, retain_graph=True)[0]
            neumann_vjp = neumann_vjp + (-1) * coeff_fn(k) * vjp
    vjp_jac = torch.autograd.grad(g, x, neumann_vjp, create_graph=training)[0]
    logdetgrad = torch.sum(vjp_jac.view(x.shape[0], -1) * vareps.view(x.shape[0], -1), 1)
    return logdetgrad

def linear_logdet_estimator(W, bs):
    n = W.shape[0]
    I = torch.eye(n, device=W.device)
    return torch.log(torch.det(I - W)) * torch.ones(bs, 1, device=W.device)

def mem_eff_wrapper(): # Function to store the gradients in the forward pass. To be implemented. 
    return 0

def geometric_sample(p, n_samples):
    return np.random.geometric(p, n_samples)

def geometric_1mcdf(p, k, offset):
    if k <= offset:
        return 1.
    else:
        k = k - offset
    """P(n >= k)"""
    return (1 - p)**max(k - 1, 0)

def poisson_sample(lamb, n_samples):
    return np.random.poisson(lamb, n_samples)

def poisson_1mcdf(lamb, k, offset):
    if k <= offset:
        return 1.
    else:
        k = k - offset
    """P(n >= k)"""
    sum = 1.
    for i in range(1, k):
        sum += lamb**i / math.factorial(i)
    return 1 - np.exp(-lamb) * sum