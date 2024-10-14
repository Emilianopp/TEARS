#!/bin/bash

model_paths=

data_name="ml-1m"  # Replace with the actual data_name variable

declare -A paths



if [ "$data_name" == "ml-1m" ]; then
    paths=(
        ["TEARS Base"]="../scratch/saved_model/${data_name}/[path-to-model]"
        ["TEARS MVAE"]="../scratch/saved_model/${data_name}/[path-to-model]"
        ["TEARS MacridVAE"]="../scratch/saved_model/${data_name}/[path-to-model]"
        ["TEARS RecVAE"]="../scratch/saved_model/${data_name}/[path-to-model]"
    )
elif [ "$data_name" == "netflix" ]; then
    paths=(
        ["TEARS RecVAE"]="../scratch/saved_model/${data_name}/[path-to-model]"
        ["TEARS MacridVAE"]="../scratch/saved_model/${data_name}/[path-to-model]"
        ["TEARS MVAE"]="../scratch/saved_model/${data_name}/[path-to-model]"
        ["TEARS Base"]="../scratch/saved_model/${data_name}/[path-to-model]"
    )
elif [ "$data_name" == "goodbooks" ]; then
    paths=(
        ["TEARS RecVAE"]="../scratch/saved_model/${data_name}/[path-to-model]"
        ["TEARS MacridVAE"]="../scratch/saved_model/${data_name}/[path-to-model]"
        ["TEARS MVAE"]="../scratch/saved_model/${data_name}/[path-to-model]"
        ["TEARS Base"]="../scratch/saved_model/${data_name}/[path-to-model]"
    )
fi

model_paths=("${paths[@]}")

model_list=$(IFS=,; echo "${paths["TEARS Base"]},${paths["TEARS MVAE"]},${paths["TEARS MacridVAE"]},${paths["TEARS RecVAE"]}")

echo "Model paths: $model_list"


declare -A vae_paths

if [ "$data_name" == "ml-1m" ]; then
    vae_paths=(
        ["TEARS MVAE"]="[model_file]"
        ["TEARS MacridVAE"]="[model_file]"
        ["TEARS RecVAE"]="[model_file]"
    )
elif [ "$data_name" == "netflix" ]; then
    vae_paths=(
        ["TEARS MVAE"]="[model_file]"
        ["TEARS MacridVAE"]="[model_file]"
        ["TEARS RecVAE"]="[model_file]"
    )
elif [ "$data_name" == "goodbooks" ]; then
    vae_paths=(
        ["TEARS MVAE"]="[model_file]"
        ["TEARS MacridVAE"]="[model_file]"
        ["TEARS RecVAE"]="[model_file]"
    )
fi

vae_model_paths=("${vae_paths[@]}")

vae_model_list=$(IFS=,; echo "${vae_model_paths[*]}")
vae_model_list=$(IFS=,; echo "${vae_paths["TEARS MVAE"]},${vae_paths["TEARS MacridVAE"]},${vae_paths["TEARS RecVAE"]}")

echo "VAE model paths: $vae_model_list"


#Uncomment to run any of the following experiments

# python -m model.eval_model \
#     --data_name=$data_name \
#     -l=$model_list \
#     -vl=$vae_model_list\
#     --llm_backbone="llama3.1-405b"\
#     --experiment="large_prompt" 

# python -m model.eval_model \
#     --data_name=$data_name \
#     -l=$model_list \
#     -vl=$vae_model_list \
#     --llm_backbone="llama3.1-405b"\
#     --experiment="large_eval"


# python -m model.eval_model \
#     --data_name=$data_name \
#     -l=$model_list \
#     -vl=$vae_model_list \
#     --llm_backbone="llama3.1-405b"\
#     --experiment="guided"


# python -m model.eval_model \
#     --data_name=$data_name \
#     -l=$model_list \
#     -vl=$vae_model_list \
#     --llm_backbone="llama3.1-405b"\
#     --experiment="small_prompt"\


# python -m model.eval_model \
#     --data_name=$data_name \
#     -l=$model_list \
#     -vl=$vae_model_list \
#     --llm_backbone="llama3.1-405b"\
#     --experiment="small_eval"\
