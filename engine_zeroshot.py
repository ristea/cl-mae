from typing import Iterable
import torch
import numpy as np


@torch.no_grad()
def compute_zeroshot_acc(model: torch.nn.Module, data_loader_train: Iterable, data_loader_eval: Iterable,
                         device: torch.device, knn_size: int):
    model.eval()

    # Compute embeddings TRAIN
    embeddings_train = []
    labels_train = []
    for data_iter_step, (samples, targets) in enumerate(data_loader_train):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            outputs = model.forward_features(samples)

        embeddings_train.append(outputs.detach().cpu())
        labels_train.append(targets.detach().cpu())

    embeddings_train = torch.cat(embeddings_train, 0)
    labels_train = torch.cat(labels_train, 0)

    # Compute embeddings VALID
    embeddings_valid = []
    labels_valid = []
    for data_iter_step, (samples, targets) in enumerate(data_loader_eval):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            outputs = model.forward_features(samples)

        embeddings_valid.append(outputs.detach().cpu())
        labels_valid.append(targets.detach().cpu())

    embeddings_valid = torch.cat(embeddings_valid, 0)
    labels_valid = torch.cat(labels_valid, 0)

    # Compute statistics
    top1 = 0.
    top_nn = 0.

    for i in range(0, len(embeddings_valid)):
        dist_norm = torch.norm(embeddings_train - embeddings_valid[i], dim=-1, p=None)
        nn_dist = torch.argsort(dist_norm)[:knn_size]
        top1 += int((torch.mode(labels_train[nn_dist], 0).values == labels_valid[i]).numpy())   ##Neelu: out of top 5 most occuring class matches the label

        if int((labels_train[torch.argsort(dist_norm)[:knn_size]] == labels_valid[i]).sum().numpy()) > 0:  ##Neelu: out of top 5 any class matches the label
            top_nn += 1

    acc1 = top1 / len(labels_valid)
    acc_nn = top_nn / len(labels_valid)
    print('* Acc@1 {top1:.3f} Acc@NN {top5:.3f}'.format(top1=acc1, top5=acc_nn))
    return {"acc1": acc1, "acc_nn": acc_nn}
