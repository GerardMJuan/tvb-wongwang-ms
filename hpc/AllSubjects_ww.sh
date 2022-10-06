#!/bin/bash
#SBATCH -J tvb_ww
#SBATCH -p high
#SBATCH -N 1
#SBATCH -o LOGS/tvb_ww.%N.%J.%u.%a.out # STDOUT
#SBATCH -e LOGS/tvb_ww.%N.%J.%u.%a.err # STDERR

njobs=13
csv=
out_dir=
in_dir=
model=wongwang

python hpc_run_all.py --njobs $njobs --csv $csv $in_dir $out_dir $model