#!/bin/bash --login
#$ -cwd
#$ -l nvidia_v100=1

#$ -pe smp.pe 8


# Latest version of CUDA
module load libs/cuda/11.7.0
module load apps/binapps/anaconda3/2022.10
source activate clihack2

python run_tr.py