# Reinforced Sequential Monte Carlo for Amortised Sampling

This repository contains the code for the paper "[Reinforced Sequential Monte Carlo for Amortised Sampling](https://arxiv.org/abs/2510.11711)" (ICML 2026 spotlight).

This repository builds upon the [repository](https://github.com/DenisBless/variational_sampling_methods) of the paper "[Beyond ELBOs: A Large-Scale Evaluation of Variational Methods for Sampling](https://arxiv.org/abs/2406.07423)" by Denis Blessing et al. (2024). We correct the GFlowNet implementation of the original repository and incorporate sequential Monte Carlo to it.

For ALDP experiments, please refer to the [PyTorch version](https://github.com/hyeok9855/ReinforcedSMC-Torch) of this project.
For the biochemical sequence design with prepend/append models, please refer to the gfn-discrete folder of the [gfn-is](https://github.com/hyeok9855/gfn-is) repository.

## Installation
- python 3.10.14
- jax 0.6.2

We recommend using the conda (or mamba) environment to install the dependencies.
```bash
conda create -n gfn-smc-jax python=3.10.14
conda activate gfn-smc-jax
```

Install tensorflow first since it sometimes causes conflicts with other packages.
```bash
pip install tensorflow==2.16.1
```

Install the jax and jaxlib with the appropriate CUDA version or TPU support, e.g., cuda12
```bash
pip install -U "jax[cuda12]==0.6.2"
```

Install the other dependencies.
```bash
pip install -r requirements.txt
```


## Usage

Here we mainly focus on the GFlowNet-based algorithms. 

Basic usage:
```bash
python run.py algorithm=<algorithm_name> target=<target_name>
```

`<algorithm_name>` can be one of the following:
- `gfn_tb` (for TB or LV loss with importance-weighted buffer (IW-Buf; section 3.3))
- `gfn_subtb_smc` (for TB/SubTB combined loss with IW-Buf and sequential Monte Carlo (SMC; section 3.2))
- `dds` (for DDS baseline)
- `pis` (for PIS baseline)
- `smc_mh` (for SMC-RWM baseline)
- `smc` (for SMC-HMC baseline)

For CMCD and SCLD baselines, please refer to the [repository of SCLD](https://github.com/anonymous3141/SCLD). While there are many other sampling methods in this repository, inherited from the [Beyond ELBOs repository](https://github.com/DenisBless/variational_sampling_methods), we have not carefully tested them in our paper.

`target_name` can be one of the following:
- Gradient-free setting
  - `gaussian_mixture40`
  - `gaussian_mixture40_5d`
  - `funnel`
  - `many_well`
- Gradient-based setting
  - `funnel_lp`
  - `planar_robot_4goals`
  - `gaussian_mixture40_50d`
  - `student_t_mixture_50d`
  - `many_well_64d`

Please refer to our paper for more details on the algorithms and targets.

## Citation

If you use parts of this repository in your work, please cite us using the following BibTeX entry:

```bibtex
@inproceedings{choi2026reinforced,
  title={Reinforced Sequential {M}onte {C}arlo for Amortised Sampling},
  author={Choi, Sanghyeok and Mittal, Sarthak and Elvira, V{\'\i}ctor and Park, Jinkyoo and Whitammer, Esmeralda S.},
  booktitle={Forty-third International Conference on Machine Learning},
  year={2026}
}
```
