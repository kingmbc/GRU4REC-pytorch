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
datasets=(yoochoose retailrocket diginetica) # (yoochoose retailrocket diginetica yoochoose1_4 yoochoose1_64)
losses=(TOP1-max BPR-max) #(TOP1 BPR TOP1-max BPR-max CrossEntropy SampledCrossEntropy)
n_epochs=(10)  #(5 10 20)
batch_sizes=(50 64) # (32 50 500)
embedding_dims=(-1 100 200)

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