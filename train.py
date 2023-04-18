'''
2023/4/16 Author: Longshen Ou
Demo of Dwivedi's Graph Transformer
Paper: https://arxiv.org/abs/2012.09699
'''
import os
import dgl
import torch
import torch.nn as nn
import torch.optim as optim
import random
import warnings
from dgl.data import AsGraphPredDataset
from dgl.data.utils import Subset
from dgl.dataloading import GraphDataLoader
from tqdm import tqdm

from models_gt import GTModelFeatDense

random.seed(42)
warnings.filterwarnings("ignore", category=UserWarning)

hparams = {
    'lr': 0.00005,
    'batch_size': 256,
    'pos_enc_size': 8,
}


def _main():
    # dev = torch.device("cpu")
    dev = torch.device("cuda:1")
    data_dir = './data/AIDS'
    dataset_path = os.path.join(data_dir, 'dataset.pt')

    # Load dataset.
    if not os.path.exists(dataset_path):
        dataset = AsGraphPredDataset(
            dgl.data.TUDataset('AIDS', raw_dir='./data/'),
            split_ratio=(0.8, 0.1, 0.1),
        )

        # Laplacian positional encoding.
        indices = torch.cat([dataset.train_idx, dataset.val_idx, dataset.test_idx])
        for idx in tqdm(indices, desc="Computing Laplacian PE"):
            g, _ = dataset[idx]
            g.ndata["PE"] = dgl.laplacian_pe(g, k=hparams['pos_enc_size'], padding=True)

        torch.save(dataset, dataset_path)
    else:
        dataset = torch.load(dataset_path)

    # Create model.
    out_size = dataset.num_tasks
    model = GTModelFeatDense(out_size=out_size, pos_enc_size=hparams['pos_enc_size']).to(dev)

    # Start training.
    train(model, dataset, dev)


def compute_acc(out, tgt):
    '''
    out and tgt are 1-d list with same length
    '''
    cor = 0
    tot = 0
    for i, j in zip(out, tgt):
        tot += 1
        if i == j:
            cor += 1
    return cor / tot


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()

    y_true = []
    y_pred = []
    for batched_g, labels in dataloader:
        batched_g, labels = batched_g.to(device), labels.to(device)
        y_hat = model(batched_g, batched_g.ndata["node_attr"], batched_g.ndata["PE"])
        y_true.append(labels.view(y_hat.shape).detach().cpu())
        y_pred.append(y_hat.detach().cpu())
    y_true = torch.cat(y_true, dim=0).squeeze()
    y_pred = torch.cat(y_pred, dim=0).squeeze()

    loss_func = nn.BCEWithLogitsLoss()
    loss = loss_func(y_pred, y_true.float())

    # Compute output
    prob = torch.sigmoid(y_pred)
    out = prob.clone()
    out[out >= 0.5] = 1
    out[out < 0.5] = 0
    out = out.long()
    acc = compute_acc(out, y_true)

    ret = {
        'loss': loss.item(),
        'acc': acc,
    }

    return ret


def train(model, dataset, device):
    train_dataloader = GraphDataLoader(
        Subset(dataset, dataset.train_idx),
        batch_size=hparams['batch_size'],
        shuffle=True,
    )
    valid_dataloader = GraphDataLoader(
        Subset(dataset, dataset.val_idx),
        batch_size=hparams['batch_size'],
    )
    test_dataloader = GraphDataLoader(
        Subset(dataset, dataset.test_idx),
        batch_size=hparams['batch_size'],
    )
    optimizer = optim.Adam(model.parameters(), lr=hparams['lr'])
    num_epochs = 20
    loss_fcn = nn.BCEWithLogitsLoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batched_g, labels in tqdm(train_dataloader):
            batched_g, labels = batched_g.to(device), labels.to(device)  # BS: 256
            logits = model(
                batched_g, batched_g.ndata["node_attr"], batched_g.ndata["PE"]  # batched_g.edata['feat'],
            )
            loss = loss_fcn(logits, labels.float())
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        avg_loss = total_loss / len(train_dataloader)
        val_metric = evaluate(model, valid_dataloader, device)
        print('Epoch: {:03d}, Loss: {:.4f}, Val loss: {:.4f}, Val acc: {:.4f}'.format(
            epoch, avg_loss, val_metric['loss'], val_metric['acc']))
    test_metric = evaluate(model, test_dataloader, device)
    print(test_metric)


if __name__ == '__main__':
    _main()
