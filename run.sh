#!/bin/bash
#SBATCH --job-name=iAF692
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --partition=compsci-gpu
#SBATCH --output=iAF692.out
#SBATCH --error=iAF692.err

data_name=iAF692
seed=0
num_epochs=2000
wd=0.0
normalize_m=1
dropout=0.0
lr=0.001
n_layers=3
hidden_channels=64
model_name="EdgeHGNAM"
patience=50
batch_size=32
train_size=0.6
val_size=0.0

python main.py --seed=$seed --wd=$wd --model_name=$model_name --data_name=$data_name --dropout=$dropout --n_layers=$n_layers --hidden_channels=$hidden_channels  --lr=$lr --num_epochs=$num_epochs --early_stop=1 --one_m=0 --normalize_m=$normalize_m --bias=1  --patience=$patience --batch_size=$batch_size --train_size=$train_size --val_size=$val_size