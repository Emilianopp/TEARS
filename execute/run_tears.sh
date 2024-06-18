#!/bin/bash

# **With a trained VAE** you can train a TEARS model
# make sure to enter the approriate path for the VAE backbone model


vae_path="ENTER PATH"
python -m trainer.train \
    --seed=2024 \
    --vae_path=$vae_path\ 
    --embedding_module="TearsRecVAE" \
    --data_name='ml-1m' \
    --dropout=.4 \
    --epochs=30 \
    --kfac=2 \
    --model_name="ot_train_vae_TearsRecVAE" \
    --lora_alpha=16 \
    --lora_r=64 \
    --bs=64 \
    --lr=.001 \
    --scheduler=None \
    --epsilon=.5 \
    --eval_control \
    --wandb
