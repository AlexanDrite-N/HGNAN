#!/bin/bash

data_name=Mushroom
runs=10
num_epochs=2000
normalize_m=1
model_name="HGNAM-node"
patience=50
train_size=0.5
val_size=0.25
lr=0.001
wd=0.0
dropout=0.5
n_layers=3
hidden_channels=64
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
  --mode=$mode
