# Data 

### MovieLens

Download the original dataset from:

[https://grouplens.org/datasets/movielens/1m/](https://grouplens.org/datasets/movielens/1m/) and place it in the `/data` directory

### Netflix

Download the original Netflix price dataset:

[https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data)

The following repo contains the genres:

[https://github.com/tommasocarraro/netflix-prize-with-genres](https://github.com/tommasocarraro/netflix-prize-with-genres)

Execute 

```bash
python -m preprocess.subset_netflix
```

### Goodbooks

Download the original datset from:

[https://www.kaggle.com/datasets/zygmunt/goodbooks-10k](https://www.kaggle.com/datasets/zygmunt/goodbooks-10k)


Get the genre information from (done automatically in code):

[https://github.com/malcolmosh/goodbooks-10k-extended/blob/master/README.md](https://github.com/malcolmosh/goodbooks-10k-extended/blob/master/README.md)


## Data pipeline

To reproduce the dataset subset simply run 

```bash
python -m preprocess.data_process_full_data --data_name=${datset_name}
```

To make the final files for training use 

```bash
python -m preprocess.make_data --data_name=${datset_name}
```

## Summaries 

To remake the summaries simply run

```bash
python -m preprocess.make_in_context_data --data_name=${datset_name}
```