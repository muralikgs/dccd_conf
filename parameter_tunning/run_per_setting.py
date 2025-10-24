import numpy as np
import os 
import argparse 
import yaml 
import pandas as pd 
import torch

from itertools import product 

from models.helper.functions import gumbelSoftMLP, gnet_z 
from models.dccd.implicitblock import imBlock 
from models.dccd.trainer_dccd import DCCDTrainer 

from utils.error_metrics import *

lambda_c_vals = [1e-3, 1e-2, 1e-1]
rho_vals = [1e-2, 1e-1, 1.0]
n_layers_vals = [0, 1, 2, 3]
lr_vals = [1e-3, 1e-2, 1e-1]

CONFIGS = list(product(lambda_c_vals, rho_vals, n_layers_vals, lr_vals))

def train_dccd(model_settings, datasets, targets, config_id):
    
    d = datasets[0].shape[1]
    param_config = CONFIGS[config_id]

    # initialize the model
    g_x = gumbelSoftMLP(
        n_nodes = d,
        lip_constant = model_settings["lip_const"],
        activation = model_settings["activation"],
        n_hidden = param_config[2]
    )
    g_z = gnet_z(n_nodes=d)
    implicit_block = imBlock(nnet_x=g_x, nnet_z=g_z, confounders=True)

    # initialize the trainer
    im_trainer = DCCDTrainer(
        implicit_block,
        max_epochs=model_settings["max_epochs"],
        batch_size=model_settings["batch_size"],
        lr=param_config[3],
        lambda_c=param_config[0],
        rho=param_config[1]
    )

    # train the model
    training_failed = False
    training_error = None
    try:
        im_trainer.train(
            intervention_datasets=datasets,
            intervention_targets=targets,
        )
    except Exception as e:
        training_failed = True
        training_error = str(e)
    
    return implicit_block, training_failed, training_error

def eval_dccd(model, datasets, targets, gt_params, model_settings):

    est_adjacency = model.nnet_x.get_w_adj().detach().cpu().numpy()
    est_covariance = model.cov_mat.detach().cpu().numpy()
    
    gt_adjacency, gt_covariance = gt_params 

    adj_thresh = model_settings["adj_threshold"]
    cov_thresh = model_settings["cov_threshold"]    

    # directional edge metrics
    shd, _ = compute_shd(np.abs(gt_adjacency) > 0, est_adjacency > adj_thresh)
    _, g_area, _, _ = compute_auprc(np.abs(gt_adjacency) > 0, est_adjacency, min_threshold=0, logspace=False)

    # bidirectional edge metrics
    diagonal_mask = np.ones_like(gt_covariance)
    np.fill_diagonal(diagonal_mask, 0)

    np.fill_diagonal(gt_covariance, 0)
    np.fill_diagonal(est_covariance, 0)

    equality_mask = ( (np.abs(gt_covariance) > 0) == (np.abs(est_covariance) > cov_thresh) )*1.0

    confounder_tp = ( equality_mask *  (np.abs(gt_covariance) > 0)*1.0 ).sum()/2
    confounder_fp = ( np.abs(est_covariance) > cov_thresh ).sum()/2 - confounder_tp 

    confounder_fn_tp = ( np.abs(gt_covariance) > 0 ).sum()/2
    confounder_fn = confounder_fn_tp - confounder_tp
    
    precision = confounder_tp / (confounder_tp + confounder_fp) if confounder_tp + confounder_fp > 0 else 1
    recall = confounder_tp / (confounder_tp + confounder_fn)
    f1score = 2*precision*recall / (precision + recall)

    _, c_area, _, _ = compute_auprc(
        np.abs(gt_covariance) > 0, 
        np.abs(est_covariance), 
        min_threshold=-4, 
        logspace=True
    )

    # validation set eval
    av_likelihood = 0
    for data, target in zip(datasets, targets):
        model.generate_current_distribution_params(intervention_targets=target)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = torch.tensor(data).to(device).float()
        mask = torch.ones_like(x)
        mask[:, target] = 0

        _, _, logpx, _ = model.loss(x, mask)
        av_likelihood += logpx.item() 
    
    av_likelihood /= len(targets)

    return shd, g_area, f1score, c_area, av_likelihood

def train_eval(model_settings, train_data, val_data, config_id, gt_params):

    train_datasets, train_targets = train_data
    model, training_failed, training_error = train_dccd(model_settings, train_datasets, train_targets, config_id)

    val_datasets, val_targets = val_data
    shd, g_area, f1score, c_area, av_likelihood = eval_dccd(model, val_datasets, val_targets, gt_params, model_settings)

    return shd, g_area, f1score, c_area, av_likelihood, training_failed, training_error

def main(settings_path, cfg_id=0, n_trials=5):

    root_dir = os.path.dirname(settings_path)

    with open(settings_path, "r", encoding="utf-8") as f:
        settings = yaml.safe_load(f)

    model_settings = settings["model"]

    results_df = pd.DataFrame()

    for trial in range(n_trials):
        print(f"{trial+1}/{n_trials}")
        artifacts_dir = os.path.join(root_dir, f"trials/trial-{trial}/artifacts")

        # load the GT SCM parameters
        scm = np.load(os.path.join(artifacts_dir, "scm_true.npz"))
        weights = scm["adjacency"]
        covariance = scm["covariance"]

        # load the data file - training
        train_data_files = np.load(os.path.join(artifacts_dir, "train_datasets.npz"), allow_pickle=True)
        train_targets = [[target.item()] for target in train_data_files["targets"]]
        train_datasets = [train_data_files[f"data_{i}"] for i, _ in enumerate(train_targets)]

        # load the data file - validation
        val_data_files = np.load(os.path.join(artifacts_dir, "val_datasets.npz"), allow_pickle=True)
        val_targets = [[t.item() for t in target] for target in val_data_files["targets"]]
        val_datasets = [train_data_files[f"data_{i}"] for i, _ in enumerate(val_targets)]

        shd, g_area, f1score, c_area, av_likelihood, training_failed, training_error = train_eval(
            model_settings, 
            (train_datasets, train_targets), 
            (val_datasets, val_targets), 
            cfg_id, 
            (weights, covariance)
        )

        row = {
            "lambda_c" : CONFIGS[cfg_id][0],
            "rho" : CONFIGS[cfg_id][1],
            "n_hidden_layers" : CONFIGS[cfg_id][2],
            "lr" : CONFIGS[cfg_id][3],
            "cfg-id" : cfg_id,
            "trial": trial,
            "di-shd" : shd,
            "di-auprc" : g_area,
            "bi-f1score" : f1score, 
            "bi-auprc" : c_area,
            "logpx" : av_likelihood
        }

        results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)
    
    parquet_path = os.path.join(root_dir, f"results.cfg.{cfg_id}.parquet")
    results_df.to_parquet(parquet_path, engine='pyarrow')

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--settings', type=str, required=True, help='Path to the settings file')
    ap.add_argument('--config_id', type=int, required=True, help='Integer denoting the config to test')
    ap.add_argument('--n_trials', type=int, default=5, help='Number of trials')
    args = ap.parse_args()

    settings_path = args.settings
    cfg_id = args.config_id
    n_trials = args.n_trials

    main(settings_path, cfg_id, n_trials)

