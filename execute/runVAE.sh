#!/bin/bash

#First execute train a VAE model with the following command

#Change the parameters as needed min_anneal = beta in MVAE paper
#Gamma is for RecVAE
#kfac is for MacridVAE

export MASTER_PORT=$(expr 10001)
export MASTER_ADDR="127.0.0.2"export MASTER_PORT=$(expr 10001 )
export MASTER_ADDR="127.0.0.2"


python -m trainer.train\
 --seed=2024\
 --embedding_module=MacridVAE\
 --data_name=ml-1m\
 --dropout=0.4\
 --epochs=100\
 --kfac=2\
 --gamma=0.0035\
 --min_anneal=0.5\
 --bs=500\
 --lr=0.001 
