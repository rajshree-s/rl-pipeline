#!/usr/bin/env just --justfile
export PATH := join(justfile_directory(), ".env", "bin") + ":" + env_var('PATH')
pem := "rl-finetuning.pem"

run:
    uv run main.py

upgrade:
    uv lock --upgrade

create_wheel:
    uv build --wheel

instance_setup:
    scp -i {{ pem }} ./instance_setup.sh {{ env_var('instance') }}:~
    ssh -i {{ pem }} {{ env_var('instance') }} "source ./instance_setup.sh"

push_wheel_instance: create_wheel
    scp -i {{ pem }} ./dist/rl_pipeline-0.1.0-py3-none-any.whl {{ env_var('instance') }}:~

ssh:
    ssh -i {{ pem }} {{ env_var('instance') }}

start_instance:
    aws ec2 start-instances --instance-ids {{ env_var('instance_id') }}

run_wheel: push_wheel_instance
    ssh -i {{ pem }} {{ env_var('instance') }} "python3.13 -m pip install --force-reinstall rl_pipeline-0.1.0-py3-none-any.whl && export hf_token={{ env_var('hf_token') }} && python3.13 -m rl_pipeline.main"

# Use this command if you have got a fresh instance and want to run from start
run_on_instance: instance_setup push_wheel_instance run_wheel

deploy_changes: push_wheel_instance run_wheel

