#!/bin/bash

sudo dnf update -y
sudo dnf groupinstall "Development Tools" -y
sudo dnf install openssl-devel bzip2-devel libffi-devel zlib-devel -y

sudo dnf install python3.13
python3.13 -m ensurepip --upgrade

ARCH=$(uname -m)
sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/amzn2023/$ARCH/cuda-amzn2023.repo
sudo dnf clean expire-cache
sudo dnf module enable -y nvidia-driver:open-dkms
sudo dnf install -y nvidia-open


# Runpod setup
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
echo >> /root/.bashrc
echo 'eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv bash)"' >> /root/.bashrc
eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv bash)"
brew install uv
uv venv --python 3.13 .venv
source .venv/bin/activate
uv pip install rl_pipeline-0.1.0-py3-none-any.whl

uv pip install "huggingface_hub[cli]"
hf auth login

cat output1.log | grep "Similarity Rouge Score So far" | wc -l