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

echo ${model}, ${dataset}, ${loss}, ${epoch}, ${batch_size}

retailrocket_path="--data_folder ../_data/retailrocket-prep --train_data retailrocket-train.csv --valid_data retailrocket-valid.csv --item2idx_dict item_idx_dict_filtered.pkl"
yoochoose_path="--data_folder ../_data/yoochoose-prep/ --train_data yoochoose-train.csv --valid_data yoochoose-valid.csv"

if [ $# == 0 ]; then
  echo "Please specify model and dataset!"
  exit 2
fi

if [ ${dataset} == "yoochoose" ]; then
  python -u main.py --loss_type ${loss} --batch_size ${batch_size} --n_epochs ${epoch} ${yoochoose_path}
elif [ ${dataset} == "retailrocket" ]; then
  python -u main.py --loss_type ${loss} --batch_size ${batch_size} --n_epochs ${epoch} ${retailrocket_path}

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
