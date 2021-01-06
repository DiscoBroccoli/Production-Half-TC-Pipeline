#!/bin/bash
#SBATCH --nodes=4
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --mem=0
#SBATCH --account=def-vermeire
#SBATCH --mail-user=jiebao995@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --time=07:00:00
#SBATCH --job-name=Half-64-TC
#SBATCH --output=%j.out

python NN.py
