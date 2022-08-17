# Prerequisites

In this section we demonstrate how to prepare an environment with PyTorch.

MMVIS works on Linux. It requires Python 3.6+, CUDA 9.2+ and PyTorch 1.5+.

```{note}
If you are experienced with PyTorch and have already installed it, just skip this part and jump to the [next section](#installation). Otherwise, you can follow these steps for the preparation.
```

**Step 0.** Download and install Miniconda from the [official website](https://docs.conda.io/en/latest/miniconda.html).

**Step 1.** Create a conda environment and activate it.

```shell
conda create --name mmvis python=3.8 -y
conda activate mmvis
```

**Step 2.** Install PyTorch following [official instructions](https://pytorch.org/get-started/locally/), e.g.

```shell
conda install pytorch==1.10.0 torchvision cudatoolkit=11.3 -c pytorch -y
```

# Installation

**Step 0.** Install [MMCV](https://github.com/open-mmlab/mmcv), [MMDet](https://github.com/open-mmlab/mmdetection) and [MMTracking](https://github.com/open-mmlab/mmtracking).

```shell
pip install mmcv-full mmdet mmtrack
```

**Step 1.** Install MMVIS.

```shell
git clone https://github.com/yingkaining/mmvis.git
cd mmvis
pip install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```

## Example conda environment setup

```shell
conda create --name mmvis python=3.8 -y
conda activate mmvis
conda install pytorch==1.10.0 torchvision==0.11.0 cudatoolkit=11.3 -c pytorch -c conda-forge -y
pip install mmcv-full==1.6.0 mmdet mmtrack

git clone https://github.com/yingkaining/mmvis.git
cd mmvis
pip install -v -e .
```
