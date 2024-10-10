# Stochastic Concept Bottleneck Models
This repository contains the code for the paper "*Stochastic Concept Bottleneck Models*" (SCBM).
(https://neurips.cc/virtual/2024/poster/94002)

**Abstract**: Concept Bottleneck Models (CBMs) have emerged as a promising interpretable method whose final prediction is based on intermediate, human-understandable concepts rather than the raw input. 
Through time-consuming manual interventions, a user can correct wrongly predicted concept values to enhance the model's downstream performance.
We propose *Stochastic Concept Bottleneck Models* (SCBMs), a novel approach that models concept dependencies. In SCBMs, a single-concept intervention affects all correlated concepts, thereby improving intervention effectiveness. Unlike previous approaches that model the concept relations via an autoregressive structure, we introduce an explicit, distributional parameterization that allows SCBMs to retain the CBMs' efficient training and inference procedure. 
Additionally, we leverage the parameterization to derive an effective intervention strategy based on the confidence region.
We show empirically on synthetic tabular and natural image datasets that our approach improves intervention effectiveness significantly. Notably, we showcase the versatility and usability of SCBMs by examining a setting with CLIP-inferred concepts, alleviating the need for manual concept annotations.

## Instructions

1. Install the packages and dependencies in the file `environment.yml`. 
2. Download the datasets described in the manuscript and update the `data_path` variable in `./configs/data/data_defaults.yaml`. For CUB, we use the original Concept Bottleneck Model's CUB version. For CIFAR, we use the concept names from Label-Free Concept Bottleneck Models. 
3. If using CIFAR-10 / CIFAR-100, run `./datasets/create_dataset_cifar.py`
4. For Weights & Biases support, set mode to 'online' and adjust entity in `./configs/config.yaml`.
5. Run the script `train.py` with the desired configuration of dataset and model from the `./configs/` folder. We provide a description of all arguments in the config files.  
**Specifying the type of SCBM**:  Replace `config.model.cov_type` and `config.model.reg_precision` to change between global and amortized version.

## Running Experiments

We provide a script in the `./scripts/` directory to run experiments on a cluster and reproduce our results. For local experimentation, we provide selected examples:

- **Amortized Variant on CUB Dataset (default)**:  
  `python train.py +model=SCBM +data=CUB`  
- **Global Variant without regularization on CUB Dataset:**:  
  `python train.py +model=SCBM +data=CUB model.cov_type='global' model.reg_precision=None`  
- **Amortized Variant on Synthetic Dataset**:  
  `python train.py +model=SCBM +data=synthetic model.encoder_arch='FCNN' model.j_epochs=150`  

## Citing
To cite SCBM please use the following BibTEX entry:

```
@inproceedings{
vandenhirtz2024stochastic,
title={Stochastic Concept Bottleneck Models},
author={Vandenhirtz, Moritz and Laguna, Sonia and Marcinkevi{\v{c}}s, Ri{\v{c}}ards and Vogt, Julia E},
booktitle={Thirty-eighth Conference on Neural Information Processing Systems},
year={2024}
url={https://openreview.net/forum?id=iSjqTQ5S1f}
}
```

