A reporsitory for replicating HyperGraph Neural Additive Network(HGNAN). HGNAN could achieve performance comparable to various baselines in both node and hyperedge prediction tasks while providing interpretations of exact decision-making process of the underlying model.

To begin with, set "train_size" and "val_size" in run.sh to "0.5" and "0.25" respectively. If you want to do node prediction tasks, set "model_name" to "HGNAM-node". For hyperedge prediction tasks, set "model_name" to "HGNAM-edge"

If you want to tuning the model, run 
```
bash tuning.sh
```

If you have a specific set of hyperparameters, run with
```
bash run_one_experiment.sh
```

Processed datasets and saved models could be found [here](https://drive.google.com/drive/folders/1Tii_EdlOwq1BprRIjV8I4Q5zXXVFHM9U?usp=drive_link). Please download them and move them into "processed_data/" if needed.
