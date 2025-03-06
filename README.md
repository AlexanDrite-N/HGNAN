A reporsitory for replicating HyperGraph Neural Additive Model(HGNAM). Paper is still under work.

This model is designed for two downstream tasks: node classification and link prediction.

To use it for node classification task, set "model_name", "train_size" and "val_size" in run.sh to "HGNAM", "0.5" and "0.25" respectively.

To use it for link prediction task, set "model_name", "train_size" and "val_size" in run.sh to "EdgeHGNAM", "0.6" and "0.0" respectively.

If you want to tuning the model, run "tuning.sh". If you have a specific set of hyperparameters, you can train the model using "run_one_experiment.py".

If you have multiple gpus and want to run them in parallel, change "python main.py" in .sh files to "python main_para.py".
