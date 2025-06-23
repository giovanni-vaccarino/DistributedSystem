# Installation

This guide is to setup up a Python 3.9 environment with PyTorch on a Raspberry Pi 4, using `pyenv` and custom wheel installation.

## 1. Install `pyenv` to Manage Python Versions

```bash
curl https://pyenv.run | bash
```

Then, add the following lines to your ~/.bashrc:

```bash
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
```

Then reload your shell:
```bash
source ~/.bashrc
```

Install required build dependencies:
```bash
sudo apt install -y make build-essential libssl-dev zlib1g-dev \
  libbz2-dev libreadline-dev libsqlite3-dev curl llvm libncursesw5-dev \
  xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
```

Install Python 3.9.13:
```bash
pyenv install 3.9.13
```

## 2.  Create and Activate a Virtual Environment with Python 3.9

```bash
pyenv virtualenv 3.9.13 fl
pyenv activate fl
```

## 3. Install Additional Dependencies

```bash
sudo apt-get install python3-pip libjpeg-dev libopenblas-dev libopenmpi-dev libomp-dev
pip install setuptools==58.3.0
pip install gdown
```

## 4. Install NumPy (Compatible Version)

```bash
pip install numpy==1.26.4 --force-reinstall
```

## 5. Download PyTorch Wheel

```bash
gdown https://drive.google.com/uc?id=1mPlhwM47Ub3SwQyufgFj3JJ9oB_wrU5D
```

## 6. Install PyTorch

```bash
pip install torch-2.0.0a0+gite9ebda2-cp39-cp39-linux_aarch64.whl
```

## 7. Install further dependencies

```bash
pip install flwr pandas
```
