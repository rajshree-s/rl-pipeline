#!/usr/bin/env just --justfile
export PATH := join(justfile_directory(), ".env", "bin") + ":" + env_var('PATH')

run:
  uv sync
  uv run main.py

upgrade:
  uv lock --upgrade
