#!/bin/bash

# **With a trained VAE** you can train a TEARS model
# make sure to enter the approriate path for the VAE backbone model
export MASTER_PORT=$(expr 10001)
export MASTER_ADDR="127.0.0.2"export MASTER_PORT=$(expr 10001 )
export MASTER_ADDR="127.0.0.2"

# vae_path="ENTER PATH"
vae_path="MacridVAE_ml-1m_2024-06-18_00-33-35_2024"
python -m trainer.train\
    --seed=2024\
    --embedding_module="TearsMacrid"\
    --data_name='ml-1m'\
    --dropout=.4\
    --epochs=1\
    --kfac=2 \
    --lora_alpha=16 \
    --lora_r=64 \
    --bs=64 \
    --lr=.001 \
    --scheduler=None \
    --epsilon=.5 \
    --eval_control \
    --wandb\
    --vae_path=$vae_path  