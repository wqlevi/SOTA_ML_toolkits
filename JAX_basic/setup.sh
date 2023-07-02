#!/usr/bin/env bash
printf "\e[33mBuidling venv for reproducibility...\n"
python -m venv ./.JAX_venv
source ./.JAX_venv/bin/activate
printf "\e[32mDone\n"
python -m pip install --upgrade pip
printf "\e[33mInstalling relevant packages...\n"
pip install -r requirement.txt
printf "\e[32mDone, ready to use\n"


