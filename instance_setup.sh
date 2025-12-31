sudo dnf update -y
sudo dnf groupinstall "Development Tools" -y
sudo dnf install openssl-devel bzip2-devel libffi-devel zlib-devel -y

sudo dnf install python3.13
python3.13 -m ensurepip --upgrade

