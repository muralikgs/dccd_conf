# Differentiable Cyclic Causal Discovery Under Unmeasured Confounders (DCCD-CONF)

This repository contains the implementation of the following paper: 

[Muralikrishnna G. Sethuraman, Faramarz Fekri. (2025). *Differentiable Cyclic Causal Discovery Under Unmeasured Confounders*. NeurIPS'25](https://arxiv.org/abs/2508.08450)

## Requirements

The code files were tested on Python version 3.14. The python library requirements are listed in `requirements.txt`. The testing codes are written on Jupyter notebook, please ensure Jupyter is installed prior to running them. Python dependencies can be installed using the following command: 
```bash
pip install -r requirements.txt
```
## Testing DCCD-CONF

In order to run ablation experiments run the following:

**Data generation**
```bash
python -m baselines.gen_ablation_settings --outdir <data_path>

python -m baselines.gen_benchmark_dataset --benchmark <data_path>
```
**Running DCCD-CONF on the generated data**
```bash
python -m run_benchmark_per_setting --settings <data_path> --outdir <data_path> --model dccd
```

Python notebooks present in the `notebooks/testing` folder also provide a short guide on how to train DCCD-CONF on various datasets. 


