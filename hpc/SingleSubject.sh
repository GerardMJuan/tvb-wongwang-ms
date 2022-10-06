#!/bin/bash
#SBATCH -J ww
#SBATCH -p high
#SBATCH -N 1
#SBATCH --sockets-per-node 1
#SBATCH --cores-per-socket 10
#SBATCH --threads-per-core 1
#SBATCH -o LOGS/ww_osc.%N.%J.%u.%a.out # STDOUT
#SBATCH -e LOGS/ww_osc.%N.%J.%u.%a.err # STDERR
#SBATCH --export=OMP_NUM_THREADS=1
center=
tr=
simlen=
model='wongwang'

path=
out_path=

python run_average.py $path $out_path $tr $simlen $model