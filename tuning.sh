#!/bin/bash
#SBATCH --job-name=NTU2012_param_tuning
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=168:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:4
#SBATCH --partition=compsci-gpu
#SBTACH --nodelist=gpu-compute5,gpu-compute6
#SBATCH --output=NTU2012_param_tuning.out
#SBATCH --error=NTU2012_param_tuning.err

data_name=NTU2012
runs=8
num_epochs=1000
normalize_m=1
model_name="HGNAM"
patience=50
train_size=0.5
val_size=0.25
weight=False
aggregation="neighbor"
tuning=True

wd_values=(0.0 0.0005)
lr_values=(0.001 0.01)
dropout_values=(0.0 0.5)
n_layers_values=(3 5)
hidden_channels_values=(32 64 128)
batch_size_values=(32 64)

for wd in "${wd_values[@]}"; do
  for lr in "${lr_values[@]}"; do
    for dropout in "${dropout_values[@]}"; do
      for n_layers in "${n_layers_values[@]}"; do
        for hidden_channels in "${hidden_channels_values[@]}"; do
          for batch_size in "${batch_size_values[@]}"; do
            echo "Running with wd=$wd, lr=$lr, dropout=$dropout, n_layers=$n_layers, hidden_channels=$hidden_channels, batch_size=$batch_size"
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
          done
        done
      done
    done
  done
done