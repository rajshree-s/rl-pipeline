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
