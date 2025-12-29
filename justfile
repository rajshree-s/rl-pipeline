#!/usr/bin/env just --justfile
export PATH := join(justfile_directory(), ".env", "bin") + ":" + env_var('PATH')
pem := "rl-finetuning.pem"
instance := "ec2-user@ec2-43-204-145-119.ap-south-1.compute.amazonaws.com"

run:
  uv sync
  uv run main.py

upgrade:
  uv lock --upgrade

deploy:
  scp -i {{pem}} ./instance_setup.sh "{{instance}}:~"
  ssh -i {{pem}} {{instance}} "bash ./instance_setup.sh"
  scp -i {{pem}} ./finetune.tar.gz "{{instance}}:~"

ssh:
  ssh -i {{pem}} {{instance}}

build:
  docker build -t finetune .

docker_run:
  docker run --cpu-shares=8 -e hf_token=$hf_token finetune:latest
