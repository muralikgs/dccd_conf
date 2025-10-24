import os, argparse, yaml 

# Data generation parameters

data_params = {
    "n_samples_per_intervention" : 500,
    "out_degree" : 2,
    "max_noise_scale" : 0.5,
    "contractive" : True,
    "n_nodes" : 10,
    "cycles" : "random",
    "beta" : 1.0,
    "confounders": 0.3, 
    "interventions" : -1,
    "val_num_targets_per_setting_min" : 2,
    "val_num_targets_per_setting_max" : 2
}

model_params = {
    "lip_const" : 0.9,
    "activation" : "tanh",
    "max_epochs" : 300,
    "batch_size" : 512,
    "glasso_iters" : 2,
    "adj_threshold" : 0.7,
    "cov_threshold" : 5e-2,
}

def main(benchmark_root):
    cfg = {
        "name" : "parameter-tunning",
        "data" : {},
        "model" : {}
    }

    for param, val in data_params.items():
        cfg["data"][param] = val 
    
    for param, val in model_params.items():
        cfg["model"][param] = val 
    
    with open(os.path.join(benchmark_root, "settings.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="./", help="Root directory to write the config files")

    args = ap.parse_args()

    benchmark_root = args.outdir

    main(benchmark_root)