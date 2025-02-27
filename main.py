import argparse
from models import *
import trainer
import datasets
import uuid
import numpy as np
import torch
import os
import copy

np.random.seed(0)
seeds = np.random.randint(low=0, high=10000, size=5)

class EarlyStopping:
    def __init__(self, patience=25, min_is_better=True):
        self.patience = patience
        self.min_is_better = min_is_better
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def reset(self):
        self.counter = 0

    def __call__(self, score):
        if self.min_is_better:
            score = -score

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

def run_exp(train_loader, val_loader, test_loader, num_features, seeds, n_layers, early_stop_flag, dropout, model_name,
            num_epochs, wd,
            hidden_channels, lr, bias, patience, loss_thresh, data_name, unique_run_id, one_m, normalize_m,
            num_classes, out_dim):

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    seeds = seeds[args.seed:args.seed + 1]
    log_file = f'logs/{unique_run_id}_{data_name}_{model_name}_training_log.txt'
    os.makedirs(os.path.dirname(log_file), exist_ok=True)  # Ensure the directory exists
    with open(log_file, 'w') as log:
        for i, seed in enumerate(seeds):
            if model_name == 'HGNAM' or model_name == 'EdgeHGNAM':
              model = HGNAM(in_channels=num_features,hidden_channels=hidden_channels,num_layers=n_layers,
                          out_channels=out_dim, dropout=dropout, bias=bias, device=device,
                          limited_m=one_m,normalize_m=normalize_m)
            param_size = 0
            for param in model.parameters():
                param_size += param.nelement() * param.element_size()
            buffer_size = 0
            for buffer in model.buffers():
                buffer_size += buffer.nelement() * buffer.element_size()

            size_all_mb = (param_size + buffer_size) / 1024 ** 2
            print('model size: {:.3f}MB'.format(size_all_mb))
            log.write('model size: {:.3f}MB\n'.format(size_all_mb))

            model.to(device)
            config = {
                'lr': lr,
                'loss': loss_type.__name__,
                'hidden_channels': hidden_channels,
                'n_conv_layers': n_layers,
                'wd': wd,
                'bias': bias,
                'dropout': dropout,
                'output_dim': out_dim,
                'num_epochs': num_epochs,
                'optimizer': optimizer_type.__name__,
                'model': model.__class__.__name__,
                'device': device.type,
                'loss_thresh': loss_thresh,
                'seed': seed,
                'data_name': data_name,
                'unique_run_id': unique_run_id,
                'early_stop_flag': early_stop_flag,
                'num_features': num_features,
                'limited_m': one_m,
                'normalize_m': normalize_m,
                'seed index ': i,
                'num_classes': num_classes,
            }

            for name, val in config.items():
                print(f'{name}: {val}')
                log.write(f'{name}: {val}\n')

            optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=wd)
            loss = loss_type()
            early_stop = EarlyStopping(patience=patience, min_is_better=True)

            best_val_acc_model_val_acc = 0

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=0.95,
                patience=8,
                min_lr=1e-7,
            )
            for epoch in range(num_epochs):
                
                print(f"Training epoch {epoch}:")
                log.write(f"Training epoch {epoch}:\n")
                train_loss, train_acc, train_auroc, train_auprc, train_recall, train_precision, train_f1,train_time = \
                    trainer.train_epoch(model, dloader=train_loader,                 
                                        optimizer=optimizer, device=device, loss_fn=loss)
                
                print("Validating on Validation Set:")
                log.write("Validating on Validation Set:\n")
                if model_name == "HGNAM":
                    val_loss, val_acc, val_auroc, val_auprc, val_rec, val_prec, val_f1, val_time = \
                        trainer.test_epoch(model, dloader=val_loader, 
                                          device=device, val_mask=True, loss_fn=loss)
                else:
                    val_loss, val_acc, val_auroc, val_auprc, val_rec, val_prec, val_f1, val_time = \
                      trainer.test_epoch(model, dloader=val_loader, 
                                        device=device, val_mask=False, loss_fn=loss)

                scheduler.step(val_loss)
                    
                if val_acc > best_val_acc_model_val_acc:
                    best_val_acc_model_val_acc = val_acc
                    torch.save(model.state_dict(),
                              f'models/{unique_run_id}_{data_name}_{model_name}_{seed}_best_val_acc.pt')
                    best_model_state = copy.deepcopy(model.state_dict()) 
                    if model_name == "HGNAM":
                      test_loss, test_acc, test_auroc, test_auprc, test_rec, test_prec, test_f1, test_time = \
                      trainer.test_epoch(model, dloader=val_loader, 
                                        device=device, val_mask=False, loss_fn=loss)

                      best_info = (f"[Best Val Updated] Epoch: {epoch:03d}"
                          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, " f"Train AUROC: {train_auroc:.4f}, Train AUPRC: {train_auprc:.4f}, "
                          f"Train Recall: {train_recall:.4f}, Train Prec: {train_precision:.4f}, "
                          f"Train F1: {train_f1:.4f}, Train Time: {train_time:.2f} sec\n"
                          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}," 
                          f"Val AUROC: {val_auroc:.4f}, Val AUPRC: {val_auprc:.4f}, "
                          f"Val Recall: {val_rec:.4f}, Val Prec: {val_prec:.4f}, Val F1: {val_f1:.4f}, Val Time: {val_time:.2f} sec\n"
                          f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}," 
                          f"Test AUROC: {test_auroc:.4f}, Test AUPRC: {test_auprc:.4f}, "
                          f"Test Recall: {test_rec:.4f}, Test Prec: {test_prec:.4f}, Test F1: {test_f1:.4f}, Test Time: {test_time:.2f} sec\n"
                          )
                      print(best_info)
                      log.write(best_info)
                    elif model_name == "EdgeHGNAM":
                        best_info = (f"[Best Val Updated] Epoch: {epoch:03d}"
                          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, " f"Train AUROC: {train_auroc:.4f}, Train AUPRC: {train_auprc:.4f}, "
                          f"Train Recall: {train_recall:.4f}, Train Prec: {train_precision:.4f}, "
                          f"Train F1: {train_f1:.4f}, Train Time: {train_time:.2f} sec\n"
                          f"Test Loss: {val_loss:.4f}, Test Acc: {val_acc:.4f}," 
                          f"Test AUROC: {val_auroc:.4f}, Test AUPRC: {val_auprc:.4f}, "
                          f"Test Recall: {val_rec:.4f}, Test Prec: {val_prec:.4f}, Test F1: {val_f1:.4f}, Test Time: {val_time:.2f} sec\n"
                          )
                        print(best_info)
                        log.write(best_info)
                    
                epoch_info = (
                    f"Epoch: {epoch:03d}, "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, " f"Train AUROC: {train_auroc:.4f}, Train AUPRC: {train_auprc:.4f}, "
                    f"Train Recall: {train_recall:.4f}, Train Prec: {train_precision:.4f}, "
                    f"Train F1: {train_f1:.4f}, Train Time: {train_time:.2f} sec "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}," 
                    f"Val AUROC: {val_auroc:.4f}, Val AUPRC: {val_auprc:.4f}, "
                    f"Val Recall: {val_rec:.4f}, Val Prec: {val_prec:.4f}, Val F1: {val_f1:.4f}, Val Time: {val_time:.2f} sec"
                )
                print(epoch_info)
                log.write(epoch_info)

                # Early stop
                early_stop(val_loss)
                if train_loss < loss_thresh:
                    print(f'loss under {loss_thresh} at epoch: {epoch}')
                    log.write(f'loss under {loss_thresh} at epoch: {epoch}\n')
                    break
                if early_stop_flag and early_stop.early_stop:
                    print(f'early stop at epoch: {epoch}')
                    log.write(f'early stop at epoch: {epoch}\n')
                    break
            
            # Final Test
            model.load_state_dict(best_model_state) 
            final_test_loss, final_test_acc, final_test_auroc, final_test_auprc, final_test_recall, final_test_prec, final_test_f1, final_test_time = trainer.test_epoch(
                model, dloader=test_loader, device=device, val_mask=False, loss_fn=loss
            )
            final_test_info = (
                  f"Final Test Loss: {final_test_loss:.4f}, Final Test Acc: {final_test_acc:.4f}, "
                  f"Final Test AUROC: {final_test_auroc:.4f}, Final Test AUPRC: {final_test_auprc:.4f}, "
                  f"Final Test Recall: {final_test_recall:.4f}, Final Test Prec: {final_test_prec:.4f}, "
                  f"Final Test F1: {final_test_f1:.4f}, Final Test time: {final_test_time:.2f} sec"
              )
            print(final_test_info)
            log.write(final_test_info)

if __name__ == '__main__':
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', dest='lr', type=float, default=0.001)
    parser.add_argument('--hidden_channels', dest='hidden_channels', type=int, default=64)
    parser.add_argument('--n_layers', dest='n_layers', type=int, default=3)
    parser.add_argument('--num_epochs', dest='num_epochs', type=int, default=1000)
    parser.add_argument('--bias', dest='bias', type=int, default=1)
    parser.add_argument('--dropout', dest='dropout', type=float, default=0.0)
    parser.add_argument('--early_stop', dest='early_stop', type=int, default=1)
    parser.add_argument('--wd', dest='wd', type=float, default=0.001)
    parser.add_argument('--data_name', dest='data_name', type=str, default='cora_ca',
                        choices = ['cora','cora_ca','Mushroom','zoo','20newsW100','NTU2012','iAF692'])
    parser.add_argument('--model_name', dest='model_name', type=str, default='HGNAM', choices = ['HGNAM','EdgeHGNAM'])
    parser.add_argument('--seed', dest='seed', type=int, default=0, choices=[0, 1, 2, 3, 4, 5])
    parser.add_argument('--one_m', dest='one_m', type=int, default=0)
    parser.add_argument('--normalize_m', dest='normalize_m', type=int, default=1)
    parser.add_argument('--patience', dest='patience', type=int, default=200)
    parser.add_argument('--train_size', dest='train_size', type=float, default=0.5)
    parser.add_argument('--val_size', dest='val_size', type=float, default=0.25)
    parser.add_argument('--m_per_feature', dest='m_per_feature', type=bool, default=False)
    parser.add_argument('--batch_size',dest='batch_size',type=int,default=64)

    args = parser.parse_args()
    loss_thresh = 0.00001
    optimizer_type = torch.optim.Adam

    train_loader, val_loader, test_loader, num_features, num_classes= datasets.get_data(data_name=args.data_name, model_name=args.model_name, train_size=args.train_size, val_size=args.val_size,batch_size=args.batch_size)

    if num_classes == 2:
        loss_type = torch.nn.BCEWithLogitsLoss
        out_dim = 1
    else:
        loss_type = torch.nn.CrossEntropyLoss
        out_dim = num_classes

    unique_run_id = uuid.uuid1()

    print(args)

    run_exp(train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, num_features=num_features,
            n_layers=args.n_layers, early_stop_flag=args.early_stop, dropout=args.dropout,
            model_name=args.model_name,
            num_epochs=args.num_epochs, wd=args.wd,
            hidden_channels=args.hidden_channels, lr=args.lr, bias=args.bias,
            patience=args.patience,
            loss_thresh=loss_thresh, seeds=seeds, data_name=args.data_name,
            unique_run_id=unique_run_id, one_m=args.one_m, normalize_m=args.normalize_m,
            num_classes=num_classes,
            out_dim=out_dim)