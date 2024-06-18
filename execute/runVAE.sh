#!/bin/bash

#First execute train a VAE model with the following command

#Change the parameters as needed min_anneal = beta in MVAE paper
#Gamma is for RecVAE
#kfac is for MacridVAE

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
