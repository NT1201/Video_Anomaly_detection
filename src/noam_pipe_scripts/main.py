from torch.utils.data import DataLoader
from learner import Learner
from loss import *
from dataset import *
import os
from sklearn import metrics
import numpy as np
import torch

# Load datasets
normal_train_dataset = Normal_Loader(is_train=1)
normal_test_dataset = Normal_Loader(is_train=0)
anomaly_train_dataset = Anomaly_Loader(is_train=1)
anomaly_test_dataset = Anomaly_Loader(is_train=0)

# Data loaders
normal_train_loader = DataLoader(normal_train_dataset, batch_size=1, shuffle=True)
normal_test_loader = DataLoader(normal_test_dataset, batch_size=1, shuffle=True)
anomaly_train_loader = DataLoader(anomaly_train_dataset, batch_size=1, shuffle=True) 
anomaly_test_loader = DataLoader(anomaly_test_dataset, batch_size=1, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model, optimizer, scheduler, loss
model = Learner(input_dim=1024, drop_p=0.6).to(device)
optimizer = torch.optim.Adagrad(model.parameters(), lr=0.001, weight_decay=0.001)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 50])
criterion = MIL

def train(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    for (normal_inputs, ), (anomaly_inputs, ) in zip(normal_train_loader, anomaly_train_loader):
        inputs = torch.cat([anomaly_inputs, normal_inputs], dim=0)  # [2, T, 1024]
        labels = torch.tensor([[1.0], [0.0]]).to(device)             # Anomaly=1, Normal=0

        inputs = inputs.to(device)
        outputs, _ = model(inputs)                                   # [2, 1]
        loss = F.binary_cross_entropy(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    print(f'Train Loss: {train_loss / len(normal_train_loader):.4f}')
    scheduler.step()

def test_abnormal(epoch):
    model.eval()
    auc = 0
    with torch.no_grad():
        for i, (data, data2) in enumerate(zip(anomaly_test_loader, normal_test_loader)):
            inputs, gts, frames = data
            inputs = inputs.to(device)
            outputs, attn_weights = model(inputs)
            outputs = outputs.squeeze().cpu().numpy()

            score_list = np.zeros(frames[0])
            step = np.round(np.linspace(0, frames[0]//16, 33)).astype(int)
            for j in range(32):
                score_list[step[j]*16:step[j+1]*16] = outputs[j]

            gt_list = np.zeros(frames[0])
            for k in range(len(gts)//2):
                s = gts[k*2]
                e = min(gts[k*2+1], frames[0])
                gt_list[s-1:e] = 1

            inputs2, gts2, frames2 = data2
            inputs2 = inputs2.to(device)
            outputs2, _ = model(inputs2)
            outputs2 = outputs2.squeeze().cpu().numpy()

            score_list2 = np.zeros(frames2[0])
            step2 = np.round(np.linspace(0, frames2[0]//16, 33)).astype(int)
            for kk in range(32):
                score_list2[step2[kk]*16:step2[kk+1]*16] = outputs2[kk]

            gt_list2 = np.zeros(frames2[0])
            score_list3 = np.concatenate((score_list, score_list2), axis=0)
            gt_list3 = np.concatenate((gt_list, gt_list2), axis=0)

            fpr, tpr, _ = metrics.roc_curve(gt_list3, score_list3, pos_label=1)
            auc += metrics.auc(fpr, tpr)

        print(f'Epoch {epoch} - AUC: {auc / len(anomaly_test_loader):.4f}')

for epoch in range(0, 75):
    train(epoch)
    test_abnormal(epoch)
