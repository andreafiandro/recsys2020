#!/bin/bash
#SBATCH --job-name=recsys2020_like_engagement
#SBATCH --mail-type=ALL
#SBATCH --mail-user=isabeau.oliveri@polito.it
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --partition=cuda
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=16
#SBATCH --output=recsys2020_like_engagement_%j.log
#SBATCH --mem-per-cpu=1024M

module load intel/python/3.6/2019.4.088
module load nvidia/cudasdk/10.1
cd ..
source recsys2020_venv/bin/activate
cd recsys2020
python3 isa_recsys2020.py ../recsys2020_training_chunk/training_chunk_0.csv Text_tokens Like_engagement_timestamp 5 8 3 0.10
