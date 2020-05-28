#!/bin/bash

# How to run
#$ bash train_script.sh GRU4Rec yoochoose TOP1-max 5 50
#$ bash train_script.sh GRU4Rec retailrocket TOP1-max 5 50


# How to kill
#

model=$1
dataset=$2
loss=$3
epoch=$4
batch_size=$5
embedding_dim=$6

echo ${model}, ${dataset}, ${loss}, ${epoch}, ${batch_size}, ${embedding_dim}

path_diginetica="--data_folder ../_data/diginetica-prep/ --train_data diginetica-train.csv --valid_data diginetica-valid.csv"
path_reddit="--data_folder ../_data/reddit-prep/ --train_data reddit-train.csv --valid_data reddit-valid.csv"
path_retailrocket="--data_folder ../_data/retailrocket-prep --train_data retailrocket-train.csv --valid_data retailrocket-valid.csv"
path_sample="--data_folder ../_data/sample-prep/ --train_data sample-train.csv --valid_data sample-valid.csv"
path_xing="--data_folder ../_data/xing-prep/ --train_data xing-train.csv --valid_data xing-valid.csv"
path_yoochoose="--data_folder ../_data/yoochoose-prep/ --train_data yoochoose-train.csv --valid_data yoochoose-valid.csv"


if [ $# == 0 ]; then
  echo "Please specify model and dataset!"
  exit 2
fi

if [ ${dataset} == "diginetica" ]; then
  python -u main.py --loss_type ${loss} --embedding_dim ${embedding_dim} --batch_size ${batch_size} --n_epochs ${epoch} ${path_diginetica}
elif [ ${dataset} == "reddit" ]; then
  python -u main.py --loss_type ${loss} --embedding_dim ${embedding_dim} --batch_size ${batch_size} --n_epochs ${epoch} ${path_reddit}
elif [ ${dataset} == "retailrocket" ]; then
  python -u main.py --loss_type ${loss} --embedding_dim ${embedding_dim} --batch_size ${batch_size} --n_epochs ${epoch} ${path_retailrocket}
elif [ ${dataset} == "sample" ]; then
  python -u main.py --loss_type ${loss} --embedding_dim ${embedding_dim} --batch_size ${batch_size} --n_epochs ${epoch} ${path_sample}
elif [ ${dataset} == "xing" ]; then
  python -u main.py --loss_type ${loss} --embedding_dim ${embedding_dim} --batch_size ${batch_size} --n_epochs ${epoch} ${path_xing}
elif [ ${dataset} == "yoochoose" ]; then
  python -u main.py --loss_type ${loss} --embedding_dim ${embedding_dim} --batch_size ${batch_size} --n_epochs ${epoch} ${path_yoochoose}

else
    echo "(Error) There is no such model or dataset"
    exit 2
fi

echo "Train is started", ${model}, ${dataset}

#nohup python -u main.py --loss_type CrossEntropy --optimizer_type Adagrad --dropout_input 0.5 --lr 0.01 --momentum 0 --batch_size 32 --n_epochs 20 --data_folder ../_data/retailrocket-prep --train_data filtered_train_sample.pkl --valid_data filtered_valid_sample.pkl > logs/gru_train_xep.out &
#nohup python -u main.py --loss_type CrossEntropy --optimizer_type Adagrad --dropout_input 0.5 --lr 0.01 --momentum 0 --batch_size 50 --n_epochs 20 --data_folder ../_data/retailrocket-prep --train_data filtered_train_sample.pkl --valid_data filtered_valid_sample.pkl > logs/gru4rec_train_xep.out &
#nohup python -u main.py --loss_type BPR --optimizer_type Adagrad --dropout_input 0.2 --lr 0.05 --momentum 0.2 --batch_size 50 --n_epochs 10 > logs/train_bpr.out &
#nohup python -u main.py --loss_type CrossEntropy --optimizer_type Adagrad --dropout_input 0 --lr 0.01 --momentum 0 --batch_size 500 --n_epochs 10 > logs/train_xe.out &

##!/bin/bash
#nohup python -u run_train.py --loss_type TOP1 --optimizer_type Adagrad --p_dropout_hidden 0.5 --lr 0.01 --momentum 0 --batch_size 50 --n_epochs 20 > logs/train_top1.out &
#nohup python -u run_train.py --loss_type BPR --optimizer_type Adagrad --p_dropout_hidden 0.2 --lr 0.05 --momentum 0.2 --batch_size 50 --n_epochs 10 > logs/train_bpr.out &
#nohup python -u run_train.py --loss_type CrossEntropy --optimizer_type Adagrad --p_dropout_hidden 0 --lr 0.01 --momentum 0 --batch_size 500 --n_epochs 10 > logs/train_xe.out &
