import os, argparse, yaml

# Data generation parameters

common_params = {
    "n_samples_per_intervention" : 500,
    "out_degree" : 2,
    "max_noise_scale" : 0.5,
    "contractive" : True,
    "val_num_targets_per_setting_min" : 2,
    "val_num_targets_per_setting_max" : 2
}

ablation_types = [
    "ablation-confounders", 
    "ablation-cycles", 
    "ablation-nonlinearity",
    "ablation-interventions",
    "ablation-nodes"
]

default_params = {
    "n_nodes" : 10,
    "cycles" : "random",
    "beta" : 1.0,
    "confounders": 0.3, 
    "interventions" : -1
}

ablation_specific_params = {
    "ablation-confounders" : {"confounders": [0.2, 0.4, 0.6, 0.8]}, 
    "ablation-cycles": {"cycles": [0, 2, 4, 6, 8]}, 
    "ablation-nonlinearity": {"beta": [0, 0.25, 0.5, 0.75, 1.0]},
    "ablation-interventions": {"interventions": list(range(11))},
    "ablation-nodes": {"n_nodes": [10, 20, 40, 80]}
}

model_params = {
    "lip_const" : 0.9,
    "activation" : "tanh",
    "max_epochs" : 300,
    "batch_size" : 512,
    "lr" : 1e-1,
    "lc" : 1e-3,
    "glasso_iters" : 2,
    "dagma_jci" : True,
    "admg_jci" : True,
    "rho" : 1e-1,
    "adj_threshold" : 0.7,
    "cov_threshold" : 5e-2,
    "llc_adj_threshold" : 0.15
}

eval_params = {
    "metrics" : ["dir-shd", "bid-f1score", "auprc"]
}

def main(benchmark_root, n_trials, base_seed):

    for abl_type in ablation_types:
        abl_dir = os.path.join(benchmark_root, abl_type)
        if not os.path.exists(abl_dir):
            os.makedirs(abl_dir)

        cfg = {
            "name" : abl_type,
            "data" : {},
            "model" : {},
            "eval" : {}
        }

        for param, val in model_params.items():
            cfg["model"][param] = val

        for param, val in common_params.items():
            cfg["data"][param] = val
        
        for param, val in eval_params.items():
            cfg["eval"][param] = val

        settings = ablation_specific_params[abl_type]
        
        for param, val in default_params.items():
            if param not in list(settings.keys())[0]:
                cfg["data"][param] = val
        
        setting, vals = list(settings.keys())[0], list(settings.values())[0]
        for val in vals: 
            setting_dir = os.path.join(abl_dir, setting+f"-{val}")
            cfg["setting"] = setting + f"-{val}"
            if not os.path.exists(setting_dir):
                os.makedirs(setting_dir)
            
            cfg["data"][setting] = val
        
            with open(os.path.join(setting_dir, "settings.yaml"), "w") as f:
                yaml.safe_dump(cfg, f, sort_keys=False)
            
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="./", help="Root directory to write the config files")
    ap.add_argument("--n_trials", type=int, default=10, help="Number of repeats")
    ap.add_argument("--base_seed", type=int, default=2025)

    args = ap.parse_args()

    benchmark_root = args.outdir 
    n_trials = args.n_trials
    base_seed = args.base_seed

    main(benchmark_root, n_trials, base_seed)
