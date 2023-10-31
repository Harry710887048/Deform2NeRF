#!/bin/sh
#提交单个作业
#SBATCH -J-name=animatable_nerf
#SBATCH --ntasks=16

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH -p A6000
#SBATCH -A gpu2002

#SBATCH --time=240:00:00
#SBATCH --output=out.job
#SBATCH --error=error.job
#SBATCH --mail-type=ALL
#SBATCH --mail-user=address

module load share/home/gpu2002/miniconda3
source activate animatable_nerf

python train_net.py --cfg_file configs/aninerf_313.yaml exp_name aninerf_313 resume True