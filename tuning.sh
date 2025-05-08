#!/bin/bash
#SBATCH --job-name=cora_tuning
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=168:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --partition=compsci-gpu

#SBATCH --output=cora_tuning.out
#SBATCH --error=cora_tuning.err

data_name=cora
num_epochs=2000
runs=10
normalize_m=1
model_name="HGNAM"
patience=50
train_size=0.5
val_size=0.25
aggregation="neighbor"
batch_size=64
mode='train'

wd_values=(0.0005)
lr_values=(0.001 0.01)
dropout_values=(0.0 0.5)
n_layers_values=(3 5)
hidden_channels_values=(64 128 256)

for wd in "${wd_values[@]}"; do
  for lr in "${lr_values[@]}"; do
    for dropout in "${dropout_values[@]}"; do
      for n_layers in "${n_layers_values[@]}"; do
        for hidden_channels in "${hidden_channels_values[@]}"; do
          echo "Running with wd=$wd, lr=$lr, dropout=$dropout, n_layers=$n_layers, hidden_channels=$hidden_channels"
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
            --tuning
            --mode=$mode
        done
      done
    done
  done
done