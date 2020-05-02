#!/bin/bash

#How to run
# $bash run_multiple_sbatch.sh cassio
# $bash run_multiple_sbatch.sh prince
if [ $# -lt 1 ]; then
  echo "Need more parameter"
  exit 2
fi

cluster=$1
models=(GRU4Rec) # (TimeRec GRU4Rec SRGNN)
datasets=(yoochoose) # (yoochoose retailrocket diginetica yoochoose1_4 yoochoose1_64)
losses=(TOP1 BPR TOP1-max BPR-max CrossEntropy)
n_epochs=(5)  #(5 10 20)
batch_sizes=(32 50) # (32 50 500)
embedding_dims=(100 200 300)

for m in ${models[@]}; do
  for d in ${datasets[@]}; do
    for l in ${losses[@]}; do
      for e in ${n_epochs[@]}; do
        for b in ${batch_sizes[@]}; do
          for emb in ${embedding_dims[@]}; do
            if [ ${cluster} == "cassio" ]; then
              sbatch --export=model=$m,dataset=$d,loss=$l,epoch=$e,batch_size=$b,embedding_dim=$emb slurm_jobs_cassio.sbatch
            elif [ ${cluster} == "prince" ]; then
              sbatch --export=model=$m,dataset=$d,loss=$l,epoch=$e,batch_size=$b,embedding_dim=$emb slurm_jobs_prince.sbatch
            fi
          done
        done
      done
    done
  done
done