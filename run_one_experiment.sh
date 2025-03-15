#!/bin/bash
#SBATCH --job-name=NTU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=168:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --partition=compsci-gpu
#SBATCH --output=NTU_para.out
#SBATCH --error=NTU_para.err

# 固定参数
data_name=NTU2012
runs=4
num_epochs=1000
normalize_m=1
model_name="HGNAM"
patience=50
train_size=0.5
val_size=0.25
weight=False
aggregation="neighbor"
tuning=True
wd=0.0
lr=0.0001
dropout=0.1
n_layers=3
hidden_channels=128
batch_size=64

echo "Running with wd=$wd, lr=$lr, dropout=$dropout, n_layers=$n_layers, hidden_channels=$hidden_channels"
python main_para.py \
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
  --weight=$weight \
  --aggregation=$aggregation \
  --tuning=$tuning
