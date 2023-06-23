#!/bin/bash

# if environment don't exists, create it
if ! conda env list | grep -q "gmflownet"; then
    conda env create -f envs/gmflownet.yml
fi
    conda env update -f envs/gmflownet.yml

# activate environment
conda activate
conda init
conda activate gmflownet

