# Web Mining Capstone Project: Group 4


## Env Setting
```
conda create -n [env name] pip
conda activate [env name]
pip install -r requirements.txt
```

## Dataset
Download dataset of Amazon Review dataset

```
cd data/amazon
wget https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFiles/Books_5.json.gz  # download review dataset
wget https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/meta_Books.json.gz  # download metadata
gzip -d meta_Books.json.gz
```
  
## Pre-train CF-RecSys (SASRec)
```
cd pre_train/sasrec
python main.py --device=cuda --dataset Books_5
```

## TextCollabRec Train
```
cd ../../
python main.py --pretrain_stage1 --rec_pre_trained_data Books_5
```


## Evaluation
Example for TextCollabRec (stage 1):
```
python evaluate.py --gpu_num 0 --k 5 --rec_pre_trained_data Books_5 --model_stage text_collab_rec --unfrozen_recsys

```
