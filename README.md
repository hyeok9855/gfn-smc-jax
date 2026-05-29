# Reinforced sequential Monte Carlo for amortised sampling

This repository contains the code for the paper "[Reinforced sequential Monte Carlo for amortised sampling](https://arxiv.org/abs/2510.11711)".

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

Full run scripts will be uploaded upon the acceptance of the paper.

## Citation

If you use parts of this repository in your work, please cite us using the following BibTeX entry:

```bibtex
@article{choi2025reinforced,
  title={Reinforced sequential {M}onte {C}arlo for amortised sampling},
  author={Choi, Sanghyeok and Mittal, Sarthak and Elvira, V{\'\i}ctor and Park, Jinkyoo and Malkin, Nikolay},
  journal={arXiv preprint arXiv:2510.11711},
  year={2025}
}
```
