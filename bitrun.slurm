#!/bin/bash
#SBATCH --partition=gpuq                    # the DGX only belongs in the 'gpu'  partition
#SBATCH --qos=gpu                           # need to select 'gpu' QoS
#SBATCH --job-name=bookmusic_TBook_bit_llama_cs798Research_job
#SBATCH --output=/scratch/dmeher/slurm_outputs/bookmusic_TBook_bit_llama_cs798Research.%j.out
#SBATCH --error=/scratch/dmeher/slurm_outputs/bookmusic_TBook_bit_llama_cs798Research.%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                 # up to 128; 
#SBATCH --gres=gpu:A100.40gb:1
#SBATCH --mem-per-cpu=80GB                 # memory per CORE; total memory is 1 TB (1,000,000 MB)
#SBATCH --export=ALL
#SBATCH --time=5-00:00:00                   # set to 1hr; please choose carefully
#SBATCH --mail-type=BEGIN,END,FAIL     # NONE,BEGIN,END,FAIL,REQUEUE,ALL,...
#SBATCH --mail-user=dmeher@gmu.edu   # Put your GMU email address here

set echo
umask 0027

# to see ID and state of GPUs assigned
nvidia-smi

#source /scratch/dmeher/custom_env/recguru_env/bin/activate
module load gnu10
module load python
source /scratch/dmeher/custom_env/llama_env/bin/activate


python embedding3_bit.py \
	--amazon_dir ../booktgt_PTUPCDR_CS798/data/mid/ \
	--overlapping_dir ./data \
	--csv_file_1 2500_llama_music_musicbook_overlapping.csv \
	--csv_file_2 2500_llama_book_musicbook_overlapping.csv \
	--pair_name musicbook

#python embedding4.py --amazon_dir ../PTUPCDR_CS798/data/mid/ --overlapping_dir ./data --csv_file_1 2500_llama_music_musicfood_overlapping.csv --csv_file_2 2500_llama_food_musicfood_overlapping.csv --pair_name musicfood
#python embedding2.py --amazon_dir ../datasets_recguru/ --overlapping_dir ./data --csv_file_1 reviews_Movies_and_TV_5.csv --csv_file_2 reviews_Grocery_and_Gourmet_Food_5.csv --pair_name movie_food

#python embedding.py
#python overlapping.py
