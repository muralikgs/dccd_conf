import numpy as np 
import os 
import argparse 
import yaml 
import uuid
import pandas as pd

from dagma.nonlinear import DagmaMLP, DagmaNonlinear

from time import time 
from tqdm import tqdm

from models.helper.functions import gumbelSoftMLP, gnet_z 
from models.dccd.implicitblock import imBlock
from models.dccd.trainer_dccd import DCCDTrainer 

from models.nodags.resblock import iResBlock
from models.nodags.trainer_nodags import NODAGSTrainer

from baselines.llc import LLCAlgorithm
from baselines.admg import *

from utils.error_metrics import * 
import torch

def train_model(data_config, model_config, datasets, targets, model_choice="dccd"):
    """
    Wrapper to dispatch training based on model_choice.
    Extra keyword args are forwarded only to the selected trainer.
    """
    if model_choice == "dccd":
        return train_dccd(data_config, model_config, datasets, targets)
    elif model_choice == "nodags":
        return train_nodags(data_config, model_config, datasets, targets)
    elif model_choice == "llc":
        return train_llc(data_config, model_config, datasets, targets)
    elif model_choice == "dagma":
        return train_dagma(data_config, model_config, datasets, targets)
    elif model_choice == "admg":
        return train_admg(data_config, model_config, datasets, targets)
    else:
        raise ValueError(f"Unknown model_choice: {model_choice}")

def train_dccd(data_config, model_config, datasets, targets):

    # initialize the model
    g_x = gumbelSoftMLP(
        n_nodes = data_config["n_nodes"],
        lip_constant = model_config["lip_const"],
        activation = model_config["activation"]
    )
    g_z = gnet_z(n_nodes=data_config["n_nodes"])
    implicit_block = imBlock(nnet_x=g_x, nnet_z=g_z, confounders=True)

    # initialize the trainer
    im_trainer = DCCDTrainer(
        implicit_block,
        max_epochs=model_config["max_epochs"],
        batch_size=model_config["batch_size"],
        lr=model_config["lr"],
        lambda_c=model_config["lc"],
        rho=model_config["rho"]
    )

    # train the model
    start = time()

    training_failed = False
    training_error = None
    try:
        im_trainer.train(
            intervention_datasets=datasets,
            intervention_targets=targets,
            verbose=verbose
        )
    except Exception as e:
        training_failed = True
        training_error = str(e)
        if verbose:
            print(f"Training failed: {training_error}")

    stop = time()

    return g_x.get_w_adj().detach().cpu().numpy(), implicit_block.cov_mat.detach().cpu().numpy(), stop-start, training_failed, training_error

def train_nodags(data_config, model_config, datasets, targets):

    # initialize the model
    causal_mech = gumbelSoftMLP(
        n_nodes = data_config["n_nodes"],
        lip_constant = model_config["lip_const"],
        activation = model_config["activation"]
    )
    nodags = iResBlock(
        func=causal_mech,
        n_power_series=5, 
        learn_full_cov=False, 
        confounders=False # Since NODAGS-Flow doesn't handle confounders
    )

    # intialize the trainer
    nf_trainer = NODAGSTrainer(
        nodags=nodags, 
        max_epochs=model_config["max_epochs"],
        batch_size=model_config["batch_size"],
        lr=model_config["lr"], 
        lambda_c=model_config["lc"]
    )

    # train the model
    start = time() 

    training_failed = False
    training_error = None
    try:
        nf_trainer.train(
            intervention_datasets=datasets,
            intervention_targets=targets,
            verbose=verbose
        )
    except Exception as e:
        training_failed = True
        training_error = str(e)
        if verbose:
            print(f"Training failed: {training_error}")

    stop = time()

    return nodags.get_w_adj(), nodags.cov_mat.detach().cpu().numpy(), stop-start, training_failed, training_error

def train_llc(data_config, model_config, datasets, targets):

    # initialize LLC
    llc = LLCAlgorithm()

    # train LLC
    start = time()

    training_failed = False 
    training_error = None

    # LLC only trains on interventional data
    targets_llc = targets[1:]
    datasets_llc = datasets[1:]
    results = llc.fit(datasets_llc, targets_llc)

    stop = time() 

    return np.abs(results.adjacency).T, results.disturbance_covariance, stop-start, training_failed, training_error

def train_dagma(data_config, model_config, datasets, targets):

    # preprocessing the data
    n, d = datasets[0].shape
    K = len(targets)

    X = np.vstack(datasets)
    
    if model_config["dagma_jci"]:
        X_jci = np.zeros((X.shape[0], d + K - 1))
        X_jci[:, :d] = X 
        X = X_jci
        for i in range(1, K):
            X[i*n:(i+1)*n, d + i-1] = 1
    
    n_vars = X.shape[1]

    # initialize and train the model
    eq_model = DagmaMLP(dims=[n_vars, 10, 1], bias=True)
    model = DagmaNonlinear(eq_model)
    start = time()
    W_est = model.fit(X, lambda1=0.02, lambda2=0.05, )
    stop = time()

    training_failed = False 
    training_error = None

    return np.abs(W_est[:d, :d]), np.zeros((d, d)), stop-start, training_failed, training_error

def train_admg(data_config, model_config, datasets, targets):

    # preprocessing the data
    n, d = datasets[0].shape
    K = len(targets)

    columns = [str(i) for i in range(d)]
    lamda = 0.05
    X = np.vstack(datasets)

    if model_config["admg_jci"]:
        lamda = 0.05
        X_jci = np.zeros((X.shape[0], d + K - 1))
        X_jci[:, :d] = X 
        X = X_jci
        for i in range(1, K):
            X[i*n:(i+1)*n, d + i-1] = 1
        
        columns += [f"C{i}" for i in range(1, K)]

    data = pd.DataFrame(X, columns=columns)

    # initialize and train the model
    learn = Discovery(lamda=lamda)
    start = time()
    best_G = learn.discover_admg(data, admg_class="bowfree", verbose=False, num_restarts=1)
    stop = time()

    endogenous = [str(i) for i in range(d)]
    di_edges = np.zeros((d, d))
    for x, y in best_G.di_edges:
        if x in endogenous and y in endogenous:
            di_edges[columns.index(x), columns.index(y)] = 1

    bi_edges = np.zeros((d, d))
    for x, y in best_G.bi_edges:
        if x in endogenous and y in endogenous:
            bi_edges[columns.index(x), columns.index(y)] = 1
            bi_edges[columns.index(y), columns.index(x)] = 1

    training_failed = False 
    training_error = None

    return di_edges, bi_edges, stop - start, training_failed, training_error

def get_metrics(gt_params, est_params, adj_thresh=0.7, cov_thresh=1e-2):

    gt_adjacency, gt_covariance = gt_params 
    est_adjacency, est_covariance = est_params 

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

    return (shd, g_area), (f1score, c_area)

def create_table_row(config, trial_id, metrics, model_choice):

    row = {
        "benchmark_name" : config["name"],
        "setting" : config["setting"],
        "trial_id" : trial_id,
        "run_id" : str(uuid.uuid4()),
        "method" : model_choice
    }

    for key, val in metrics.items():
        row[key] = val

    return row 

def run_setting(data_root_dir, benchmark_root_dir, n_trials=10, verbose=False, model_choice="dccd"):
    
    with open(os.path.join(data_root_dir, "settings.yaml"), "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    model_config = config["model"]
    data_config = config["data"]

    parquet_path = os.path.join(benchmark_root_dir, f"results.{config['name']}.{config['setting']}.{model_choice}.parquet")
    results_df = pd.DataFrame()

    for trial in range(n_trials):
        print(f"Trial: {trial+1}/{n_trials}")
        artifacts_dir = os.path.join(data_root_dir, f"trials/trial-{trial}/artifacts")

        # load the GT SCM parameters        
        scm = np.load(os.path.join(artifacts_dir, "scm_true.npz"))
        weights = scm["adjacency"]
        covariance = scm["covariance"]

        # load the data files
        data_files = np.load(os.path.join(artifacts_dir, "train_datasets.npz"), allow_pickle=True)
        targets = [[target.item()] for target in data_files["targets"]]
        datasets = [data_files[f"data_{i}"] for i, _ in enumerate(targets)]

        # estimated graph, estimated covariance and total training time
        est_graph, est_cov, train_time, training_failed, training_error = train_model(
            data_config, 
            model_config, 
            datasets, 
            targets,
            model_choice
        )

        # save the estimated graph and covariance
        np.savez(os.path.join(artifacts_dir, f"scm_est.{model_choice}.npz"), adjacency=est_graph, covariance=est_cov)

        # evaluation
        di_edge_metrics, bi_edge_metrics = get_metrics(
            (weights, covariance),
            (est_graph, est_cov),
            adj_thresh=model_config["adj_threshold"] if model_choice in ["dccd", "nodags"] else model_config["llc_adj_threshold"],
            cov_thresh=model_config["cov_threshold"]
        )

        metrics = {
            "di_shd" : di_edge_metrics[0],
            "di_auprc" : di_edge_metrics[1],
            "bi_f1score" : bi_edge_metrics[0],
            "bi_auprc" : bi_edge_metrics[1],
            "training_time" : train_time,
            "training_failed" : training_failed,
            "training_error" : training_error if training_failed else ""
        }

        result_row = create_table_row(config, trial, metrics, model_choice)
        results_df = pd.concat([results_df, pd.DataFrame([result_row])], ignore_index=True)
    
    results_df.to_parquet(parquet_path, engine="pyarrow")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", type=str, help='path to the settings file')
    ap.add_argument("--root_out", type=str, help="path to the root folder to store the results")
    ap.add_argument("--n_trials", type=int, default=10, help="number of repeated trials")
    ap.add_argument("--model", type=str, default="dccd", choices=["dccd", "nodags", "llc", "dagma", "admg"], help="Model to test")
    ap.add_argument("--verbose", action="store_true", default=False, help="Use the flag for printing the loss during training")
    
    args = ap.parse_args()
    
    settings_path = args.settings
    data_root_dir = os.path.dirname(settings_path)
    benchmark_root_dir = args.root_out
    verbose = args.verbose
    n_trials = args.n_trials
    model_choice = args.model

    run_setting(data_root_dir, benchmark_root_dir, n_trials, verbose, model_choice)
