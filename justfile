#!/usr/bin/env just --justfile
export PATH := join(justfile_directory(), ".env", "bin") + ":" + env_var('PATH')
pem := "rl-finetuning.pem"
instance := "ec2-user@ec2-13-233-149-76.ap-south-1.compute.amazonaws.com"
instance_id := "i-0cc51ff5aad674078"

run:
  uv sync
  uv run main.py

upgrade:
  uv lock --upgrade

build:
  docker build -t finetune .

save_build_image:
  docker save finetune:latest | gzip > finetune.tar.gz

instance_setup:
  scp -i {{pem}} ./instance_setup.sh "{{instance}}"
  ssh -i {{pem}} {{instance}} "bash ./instance_setup.sh"

push_wheel_instance:
  scp -i {{pem}} ./dist/rl_pipeline-0.1.0-py3-none-any.whl "{{instance}}:~/"

deploy:
  scp -i {{pem}} ./finetune.tar.gz "{{instance}}"

ssh:
  ssh -i {{pem}} {{instance}}

docker_run:
  docker run --cpu-shares=8 -e hf_token=$hf_token finetune:latest

start_instance:
  aws ec2 start-instances --instance-ids {{instance_id}}

run_on_instance: build save_build_image deploy
