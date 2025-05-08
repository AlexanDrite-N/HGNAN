#!/bin/bash
#SBATCH --job-name=NTU2012
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --partition=compsci-gpu
#SBATCH --output=NTU2012.out
#SBATCH --error=NTU2012.err

data_name=NTU2012
runs=10
num_epochs=2000
normalize_m=1
model_name="HGNAM"
patience=50
train_size=0.5
val_size=0.25
aggregation="neighbor"
lr=0.001
wd=0.0
dropout=0.5
n_layers=3
hidden_channels=256
batch_size=64
mode='evaluation'

echo "Running with wd=$wd, lr=$lr, dropout=$dropout, n_layers=$n_layers, hidden_channels=$hidden_channels, batch_size=$batch_size"
python main.py \
  --runs=$runs \
  --wd=$wd \
  --model_name=$model_name \
  --data_name=$data_name \
  --dropout=$dropout \
  --n_layers=$n_layers \
  --hidden_channels=$hidden_channels  \
  --lr=$lr \
  --num_epochs=$num_epochs \
  --early_stop=1 \
  --one_m=0 \
  --normalize_m=$normalize_m \
  --bias=1  \
  --patience=$patience \
  --batch_size=$batch_size \
  --train_size=$train_size \
  --val_size=$val_size \
  --weight \
  --aggregation=$aggregation \
  --mode=$mode
