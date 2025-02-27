import torch
import time
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from utils import *

def get_accuracy(outputs, labels):
    if outputs.dim() >= 2 and outputs.shape[-1] > 1:
      return get_multiclass_accuracy(outputs, labels)
    else:
      outputs = torch.sigmoid(outputs)
      pred = torch.where(outputs > 0.5, 1, torch.where(outputs <= 0.5, 0, 0))
      return (labels == pred).sum().item()

def get_multiclass_accuracy(outputs, labels):
    assert outputs.size(1) >= labels.max().item() + 1
    probas = torch.softmax(outputs, dim=-1)
    preds = torch.argmax(probas, dim=-1)
    correct = (preds == labels).sum()
    acc = correct
    return acc

def train_epoch(model, dloader, loss_fn, optimizer, device, label_index=0):
    start_time = time.time()
    running_loss = 0.0
    n_samples = 0
    running_acc = 0.0
    all_outputs = []
    all_labels = []

    for i, data in enumerate(dloader):

        if len(data['y'].squeeze(0).shape) > 1:
            labels = data['y'].squeeze(0)[:, label_index].view(-1, 1).flatten()
            labels = labels.float()
        else:
            labels = data['y'].flatten()
        if -1 in labels:
            labels = (labels + 1) / 2
        if loss_fn.__class__.__name__ == 'CrossEntropyLoss':
            labels = labels.long()

        train_edge_mask = data['train_mask'].squeeze(0)

        data = data.to(device)
        labels = labels.to(device)

        outputs = model.forward(data)
        outputs = outputs.squeeze(-1)
        outputs = outputs[train_edge_mask]
        labels = labels[train_edge_mask]

        if isinstance(outputs, tuple):
            outputs = outputs[0]

        n_samples += len(labels)
        if outputs.dim() == 2 and outputs.shape[-1] == 1:
            loss = loss_fn(outputs.flatten(), labels.float())
        else:
            loss = loss_fn(outputs, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        with torch.no_grad():
            running_acc += get_accuracy(outputs.detach(), labels.detach())
            all_outputs.append(outputs.detach().cpu())
            all_labels.append(labels.detach().cpu())

    avg_loss = running_loss / len(dloader)
    avg_acc = running_acc / n_samples

    all_outputs_tensor = torch.cat(all_outputs, dim=0)  # [N]
    all_labels_tensor = torch.cat(all_labels, dim=0)        # [N]
    all_labels_np = all_labels_tensor.numpy()

    with torch.no_grad():
        if loss_fn.__class__.__name__ == 'CrossEntropyLoss':
            prob = torch.softmax(all_outputs_tensor, dim=-1)
            pred = torch.argmax(prob, dim=-1)
            prob_np = prob.numpy()
            pred_np = pred.numpy()
            try:
                auroc = roc_auc_score(all_labels_np, prob_np, multi_class='ovr')
            except Exception as e:
                auroc = float('nan')
            try:
                auprc = average_precision_score(all_labels_np, prob_np, average='macro')
            except Exception as e:
                auprc = float('nan')
            rec = recall_score(all_labels_np, pred_np, average='macro', zero_division=0)
            prec = precision_score(all_labels_np, pred_np, average='macro', zero_division=0)
            f1 = f1_score(all_labels_np, pred_np, average='macro', zero_division=0)
        else: 
            prob = torch.sigmoid(all_outputs_tensor)
            pred = (prob > 0.5).long()
            prob_np = prob.numpy()
            pred_np = pred.numpy()
            auprc = average_precision_score(all_labels_np, prob_np)
            rec = recall_score(all_labels_np, pred_np, zero_division=0)
            prec = precision_score(all_labels_np, pred_np, zero_division=0)
            f1 = f1_score(all_labels_np, pred_np, zero_division=0)
            auroc = roc_auc_score(all_labels_np, prob_np)

    end_time = time.time()
    duration = end_time - start_time
    print(f"[Train] Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}, "
          f"AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}, Recall: {rec:.4f}, Precision: {prec:.4f}, F1: {f1:.4f}, Time: {duration:.2f} sec")

    return avg_loss, avg_acc, auroc, auprc, rec, prec, f1, duration

def test_epoch(model, dloader, loss_fn, device, label_index=0, val_mask=True):
    start_time = time.time()
    with torch.no_grad():
        running_loss = 0.0
        n_samples = 0
        running_acc = 0.0
        model.eval()

        all_outputs = []
        all_labels_store = []
        for i, data in enumerate(dloader):
            if len(data['y'].squeeze(0).shape) > 1:
                labels = data['y'].squeeze(0)[:, label_index].view(-1, 1).flatten()
                labels = labels.float()
            else:
                labels = data['y'].flatten()  # What we use

            if val_mask:
                edge_mask = data['val_mask'].squeeze(0)
            else:
                edge_mask = data['test_mask'].squeeze(0)

            if -1 in labels:
                labels = (labels + 1) / 2
            if loss_fn.__class__.__name__ == 'CrossEntropyLoss':
                labels = labels.long()

            data = data.to(device)
            labels = labels.to(device)

            # forward
            outputs = model.forward(data)
            outputs = outputs.squeeze(-1)
            outputs = outputs[edge_mask]
            labels = labels[edge_mask]

            if isinstance(outputs, tuple):
                outputs = outputs[0]

            n_samples += len(labels)

            if outputs.dim() == 2 and outputs.shape[-1] == 1:
                loss = loss_fn(outputs.flatten(), labels.float())
            else:
                loss = loss_fn(outputs, labels)
            running_loss += loss.item()

            running_acc += get_accuracy(outputs, labels)
            all_outputs.append(outputs.detach().cpu())
            all_labels_store.append(labels.detach().cpu())

        avg_loss = running_loss / len(dloader)
        avg_acc = running_acc / n_samples

        all_outputs_tensor = torch.cat(all_outputs, 0)   # [N]
        all_labels_tensor = torch.cat(all_labels_store, 0)  # [N]
        all_labels_np = all_labels_tensor.numpy()
        
        if loss_fn.__class__.__name__=='CrossEntropyLoss':
            prob = torch.softmax(all_outputs_tensor, dim=-1)
            pred = torch.argmax(prob, dim=-1)
            prob_np = prob.numpy()
            pred_np = pred.numpy()
            try:
                auroc = roc_auc_score(all_labels_np, prob_np, multi_class='ovr')
            except Exception as e:
                auroc = float('nan')
            try:
                auprc = average_precision_score(all_labels_np, prob_np, average='macro')
            except Exception as e:
                auprc = float('nan')
            rec = recall_score(all_labels_np, pred_np, average='macro', zero_division=0)
            prec = precision_score(all_labels_np, pred_np, average='macro', zero_division=0)
            f1 = f1_score(all_labels_np, pred_np, average='macro', zero_division=0)

        else: 
            prob = torch.sigmoid(all_outputs_tensor)
            pred = (prob > 0.5).long()
            prob_np = prob.numpy()
            pred_np = pred.numpy()

            auprc = average_precision_score(all_labels_np, prob_np)
            rec = recall_score(all_labels_np, pred_np, zero_division=0)
            prec = precision_score(all_labels_np, pred_np, zero_division=0)
            f1 = f1_score(all_labels_np, pred_np, zero_division=0)
            auroc = roc_auc_score(all_labels_np, prob_np)
        end_time = time.time()
        duration = end_time - start_time
        if val_mask:
            print(f"[Validation] Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}, "
                  f"AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}, Recall: {rec:.4f}, Precision: {prec:.4f}, F1: {f1:.4f}, Time: {duration:.2f} sec")
        else:
            print(f"[Test] Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}, "
                  f"AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}, Recall: {rec:.4f}, Precision: {prec:.4f}, F1: {f1:.4f}, Time: {duration:.2f} sec")

        return avg_loss, avg_acc, auroc, auprc, rec, prec, f1, duration