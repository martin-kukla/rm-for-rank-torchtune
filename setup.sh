#! /bin/bash
pip install torchtune
pip install bitsandbytes
pip install wandb

# jupyter
pip install jupyter
/usr/local/bin/jupyter-notebook --port=3000 --ip=0.0.0.0 --no-browser --allow-root --ServerApp.token= --notebook-dir=/efs/notebooks/mkukla > /root/nb.out 2> /root/nb.err &
