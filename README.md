# MMVIS

Built upon OpenMMLab projects.

## Installation

```shell
conda create -n mmvis python=3.8 -y
conda activate mmvis

conda install pytorch=1.10.0 torchvision cudatoolkit=11.3 -c pytorch -y
pip install openmim
mim install mmcv-full
pip install -r requirements.txt
python setup.py develop
```