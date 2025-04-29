#!/bin/bash
printf "\n*** setting up environment ***\n"

# Initialize conda for the shell script
source "$(conda info --base)/etc/profile.d/conda.sh"

# deactivate any active conda environment or .venv
conda deactivate 2>/dev/null || true

# Define the environment name as a variable
ENV_NAME="pdf_anomaly"
PYTHON_VERSION="3.9"

# check if the environment exists and remove it
conda env list | grep $ENV_NAME
if [ $? -eq 0 ]; then
    printf "\n*** environment $ENV_NAME exists, removing ***\n"
    conda remove --name $ENV_NAME --all -y
fi

# create + activate conda env.
printf "\n*** creating conda env: $ENV_NAME ***\n"
conda create -n $ENV_NAME python=$PYTHON_VERSION -y
conda activate $ENV_NAME

# install core dependencies with conda
conda install -c conda-forge numpy scipy matplotlib scikit-learn pandas statsmodels pytz cython xlrd openpyxl hdf5 pytables pyarrow -y
pip install -r requirements.txt

printf "\n *** run <conda activate $ENV_NAME> to activate the environment (unless ran .sh with <source pdf_anomaly_setup.sh>) ***\n"