# Reinforced Sequential Monte Carlo for Amortised Sampling

This repository contains the code for the paper "[Reinforced Sequential Monte Carlo for Amortised Sampling](https://arxiv.org/abs/2510.11711)" (ICML 2026 spotlight).

This repository builds upon the [repository](https://github.com/DenisBless/variational_sampling_methods) of the paper "[Beyond ELBOs: A Large-Scale Evaluation of Variational Methods for Sampling](https://arxiv.org/abs/2406.07423)" by Denis Blessing et al. (2024). We correct the GFlowNet implementation of the original repository and incorporate sequential Monte Carlo to it.

For ALDP experiments, please refer to the [PyTorch version](https://github.com/hyeok9855/ReinforcedSMC-Torch) of this project.
For the biochemical sequence design with prepend/append models, please refer to the gfn-discrete folder of the [gfn-is](https://github.com/hyeok9855/gfn-is) repository.

## Installation

We recommend using [uv](https://github.com/astral-sh/uv) to install dependencies and run the project.

First, synchronize the dependencies to set up a virtual environment. Specify the CUDA version you want to use, e.g., for CUDA 12, run:
```bash
uv sync --extra cuda12
```

This will automatically create a virtual environment (`.venv`) and install all required packages (including JAX and TensorFlow) with matching versions.

## Usage

Basic usage:
```bash
uv run python run.py algorithm=<algorithm_name> target=<target_name>
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

To change the configs, you can either edit the files in the `configs` directory or override them using the command line. For example, to use LV (instead of TB) loss without the buffer:
```bash
uv run python run.py algorithm=gfn_tb target=gaussian_mixture40 algorithm.loss_type=lv algorithm.use_buffer=false
```

Please refer to our paper for more details on the algorithms and targets. There are some additional features that are not included in the paper, e.g., Pinned Brownian motion (instead of OU) as the reference process, and MALA as a rejuvenation step in SMC for `gfn_subtb_smc`.

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
