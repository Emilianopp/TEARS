# TEARS

Text Representations for Scrutable Recommendations (link to come)


![Description](./figures/tears.svg)


## Data 
We provide detailed instructions on how to reproduce the datasets used in the [data](data/README.md) directory

## User Summaries

We provide user summaries in `./saved_user_summaries`. These can be remade with 

```bash
python -m preprocess.make_in_context_data --data_name=${data_name}
```

To simply use the preprocessed data directly first run:

```bash
tar xvzf data_preprocessed.tar.gz
tar xvzf saved_user_summary.tar.gz
```

`tar xvzf data.tar.gz` contains all original data, which can be redownloaded using the instructions provided in [data](data/README.md), but we include here for convinience. 

`tar xvzf data_preprocessed.tar.gz` contains all relevant processed data that is needed to run the code.


`saved_user_summary.tar.gz` contains all LLaMA and GPT user summaries, as well as augmented summaries used to evaluate large-scope and fine-grained controllability. 



## Execution 
We provide an example execution command below, but note other examples given in `/execute`

We implement `EASE`,`MVAE`,`MDAE`, `MacridVAE`,`RecVAE` and their appropriate TEARS models

To execute the training pipeline run:

```bash
python -m trainer.train \
    --seed=2024 \
    --embedding_module=${module} \
    --data_name=${dataset} \
    --dropout=.4 \
    --epochs=30 \
    --lora_alpha=16 \
    --lora_r=64 \
    --bs=64 \
    --lr=.0001 \
    --scheduler=None \
    --epsilon=.5 \
    --eval_control \
    --wandb
```

## Requirements

## Acknowledgements

We thank the authors of the following repositories for their useful codebases that where a key role in the execution of this project: 

`MVAE/MDAE` [https://github.com/younggyoseo/vae-cf-pytorch](https://github.com/younggyoseo/vae-cf-pytorch)

`RecVAE` [https://github.com/ilya-shenbin/RecVAE](https://github.com/ilya-shenbin/RecVAE)

`MacridVAE` (adapted to torch from) [https://jianxinma.github.io/disentangle-recsys.html](https://jianxinma.github.io/disentangle-recsys.html)
