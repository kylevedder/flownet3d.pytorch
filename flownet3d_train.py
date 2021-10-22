from flownet3d import FlowNet3D

import glob
import json
import matplotlib.pyplot as plt
from multiprocessing import Manager
import numpy as np
import os
import time
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import kaolin as kal
from kaolin.models.PointNet2 import furthest_point_sampling
from kaolin.models.PointNet2 import fps_gather_by_index
from kaolin.models.PointNet2 import ball_query
from kaolin.models.PointNet2 import group_gather_by_index
from kaolin.models.PointNet2 import three_nn
from kaolin.models.PointNet2 import three_interpolate

class SceneflowDataset(Dataset):
    def __init__(self, npoints=2048, root='/Datasets/flyingthings3d', train=True, cache=None):
        self.npoints = npoints
        self.train = train
        self.root = root
        if self.train:
            self.datapath = glob.glob(os.path.join(self.root, 'TRAIN*.npz'))
        else:
            self.datapath = glob.glob(os.path.join(self.root, 'TEST*.npz'))
        
        if cache is None:
            self.cache = {}
        else:
            self.cache = cache
        
        self.cache_size = 30000

        ###### deal with one bad datapoint with nan value
        self.datapath = [d for d in self.datapath if 'TRAIN_C_0140_left_0006-0' not in d]
        ######

    def __getitem__(self, index):
        if index in self.cache:
            pos1, pos2, color1, color2, flow, mask1 = self.cache[index]
        else:
            fn = self.datapath[index]
            with open(fn, 'rb') as fp:
                data = np.load(fp)
                pos1 = data['points1'].astype('float32')
                pos2 = data['points2'].astype('float32')
                color1 = data['color1'].astype('float32')
                color2 = data['color2'].astype('float32')
                flow = data['flow'].astype('float32')
                mask1 = data['valid_mask1']

            if len(self.cache) < self.cache_size:
                self.cache[index] = (pos1, pos2, color1, color2, flow, mask1)

        if self.train:
            n1 = pos1.shape[0]
            sample_idx1 = np.random.choice(n1, self.npoints, replace=False)
            n2 = pos2.shape[0]
            sample_idx2 = np.random.choice(n2, self.npoints, replace=False)

            pos1 = pos1[sample_idx1, :]
            pos2 = pos2[sample_idx2, :]
            color1 = color1[sample_idx1, :]
            color2 = color2[sample_idx2, :]
            flow = flow[sample_idx1, :]
            mask1 = mask1[sample_idx1]
        else:
            pos1 = pos1[:self.npoints, :]
            pos2 = pos2[:self.npoints, :]
            color1 = color1[:self.npoints, :]
            color2 = color2[:self.npoints, :]
            flow = flow[:self.npoints, :]
            mask1 = mask1[:self.npoints]

        pos1_center = np.mean(pos1, 0)
        pos1 -= pos1_center
        pos2 -= pos1_center
        
        pos1 = torch.from_numpy(pos1).t()
        pos2 = torch.from_numpy(pos2).t()
        color1 = torch.from_numpy(color1).t()
        color2 = torch.from_numpy(color2).t()
        flow = torch.from_numpy(flow).t()
        mask1 = torch.from_numpy(mask1)

        return pos1, pos2, color1, color2, flow, mask1

    def __len__(self):
        return len(self.datapath)
    
train_set = SceneflowDataset(train=True)
points1, points2, color1, color2, flow, mask1 = train_set[0]

print(points1.shape, points1.dtype)
print(points2.shape, points2.dtype)
print(color1.shape, color1.dtype)
print(color2.shape, color2.dtype)
print(flow.shape, flow.dtype)
print(mask1.shape, mask1.dtype)

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def pdist2squared(x, y):
    xx = (x**2).sum(dim=1).unsqueeze(2)
    yy = (y**2).sum(dim=1).unsqueeze(1)
    dist = xx + yy - 2.0 * torch.bmm(x.permute(0, 2, 1), y)
    dist[dist != dist] = 0
    dist = torch.clamp(dist, 0.0, np.inf)
    return dist

def parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class ClippedStepLR(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, step_size, min_lr, gamma=0.1, last_epoch=-1):
        self.step_size = step_size
        self.min_lr = min_lr
        self.gamma = gamma
        super(ClippedStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [max(base_lr * self.gamma ** (self.last_epoch // self.step_size), self.min_lr)
                for base_lr in self.base_lrs]
    
def criterion(pred_flow, flow, mask):
    loss = torch.mean(mask * torch.sum((pred_flow - flow) * (pred_flow - flow), dim=1) / 2.0)
    return loss

def error(pred, labels, mask):
    pred = pred.permute(0,2,1).cpu().numpy()
    labels = labels.permute(0,2,1).cpu().numpy()
    mask = mask.cpu().numpy()
    
    err = np.sqrt(np.sum((pred - labels)**2, 2) + 1e-20)

    gtflow_len = np.sqrt(np.sum(labels*labels, 2) + 1e-20) # B,N
    acc050 = np.sum(np.logical_or((err <= 0.05)*mask, (err/gtflow_len <= 0.05)*mask), axis=1)
    acc010 = np.sum(np.logical_or((err <= 0.1)*mask, (err/gtflow_len <= 0.1)*mask), axis=1)

    mask_sum = np.sum(mask, 1)
    acc050 = acc050[mask_sum > 0] / mask_sum[mask_sum > 0]
    acc050 = np.mean(acc050)
    acc010 = acc010[mask_sum > 0] / mask_sum[mask_sum > 0]
    acc010 = np.mean(acc010)

    epe = np.sum(err * mask, 1)[mask_sum > 0] / mask_sum[mask_sum > 0]
    epe = np.mean(epe)
    return epe, acc050, acc010


# parameters
BATCH_SIZE = 16
NUM_POINTS = 2048
NUM_EPOCHS = 150
INIT_LR = 0.001
MIN_LR = 0.00001
STEP_SIZE_LR = 10
GAMMA_LR = 0.7
INIT_BN_MOMENTUM = 0.5
MIN_BN_MOMENTUM = 0.01
STEP_SIZE_BN_MOMENTUM = 10
GAMMA_BN_MOMENTUM = 0.5

# data
train_manager = Manager()
train_cache = train_manager.dict()
train_dataset = SceneflowDataset(npoints=NUM_POINTS, train=True, cache=train_cache)
train_loader = DataLoader(train_dataset,
                          batch_size=BATCH_SIZE,
                          num_workers=4,
                          shuffle=True,
                          pin_memory=True,
                          drop_last=True)
print('train:', len(train_dataset), '/', len(train_loader))

test_manager = Manager()
test_cache = test_manager.dict()
test_dataset = SceneflowDataset(npoints=NUM_POINTS, train=False, cache=test_cache)
test_loader = DataLoader(test_dataset,
                        batch_size=BATCH_SIZE,
                        num_workers=4,
                        pin_memory=True,
                        drop_last=True)
print('test:', len(test_dataset), '/', len(test_loader))

# net
def init_weights(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1.0)
        m.bias.data.fill_(0.0)

net = FlowNet3D().cuda()
net.apply(init_weights)
print('# parameters: ', parameter_count(net))

# optimizer
optimizer = optim.Adam(net.parameters(), lr=INIT_LR)

# learning rate scheduler
lr_scheduler = ClippedStepLR(optimizer, STEP_SIZE_LR, MIN_LR, GAMMA_LR)

# batch norm momentum scheduler
def update_bn_momentum(epoch):
    for m in net.modules():
        if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
            m.momentum = max(INIT_BN_MOMENTUM * GAMMA_BN_MOMENTUM ** (epoch // STEP_SIZE_BN_MOMENTUM), MIN_BN_MOMENTUM)

# statistics
losses_train = []
losses_test = []

# for num_epochs
for epoch in range(NUM_EPOCHS):
    
    # update batch norm momentum
    update_bn_momentum(epoch)
    
    # train mode
    net.train()
    
    # statistics
    running_loss = 0.0
    torch.cuda.synchronize()
    start_time = time.time()
    
    # for each mini-batch
    for points1, points2, features1, features2, flow, mask1 in train_loader:
        # to GPU
        points1 = points1.cuda(non_blocking=True)
        points2 = points2.cuda(non_blocking=True)
        features1 = features1.cuda(non_blocking=True)
        features2 = features2.cuda(non_blocking=True)
        flow = flow.cuda(non_blocking=True)
        mask1 = mask1.cuda(non_blocking=True)
        
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward + backward + optimize
        pred_flow = net(points1, points2, features1, features2)
        loss = criterion(pred_flow, flow, mask1)
        loss.backward()
        optimizer.step()
        
        # statistics
        running_loss += loss.item()
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    running_loss /= (len(train_loader))
    
    losses_train.append(running_loss)
    
    # output
    print('Epoch {} (train) -- loss: {:.6f} -- duration (epoch/iteration): {:.4f} min/{:.4f} sec'.format(epoch, running_loss, (end_time-start_time)/60.0, (end_time-start_time)/len(train_loader)))
    
    # validate
    with torch.no_grad():
      
        # eval mode
        net.eval()

        # statistics
        running_loss = 0.0
        torch.cuda.synchronize()
        start_time = time.time()
        
        # for each mini-batch
        for points1, points2, features1, features2, flow, mask1 in test_loader:
            
            # to GPU
            points1 = points1.cuda(non_blocking=True)
            points2 = points2.cuda(non_blocking=True)
            features1 = features1.cuda(non_blocking=True)
            features2 = features2.cuda(non_blocking=True)
            flow = flow.cuda(non_blocking=True)
            mask1 = mask1.cuda(non_blocking=True)

            # forward
            pred_flow = net(points1, points2, features1, features2)
            loss = criterion(pred_flow, flow, mask1)

            # statistics
            running_loss += loss.item()
            
        torch.cuda.synchronize()
        end_time = time.time()

        running_loss /= len(test_loader)

        losses_test.append(running_loss)

        # output
        print('Epoch {} (test) -- loss: {:.6f} -- duration (epoch/iteration): {:.4f} min/{:.4f} sec'.format(epoch, running_loss, (end_time-start_time)/60.0, (end_time-start_time)/len(train_loader)))
        
    # update learning rate
    lr_scheduler.step()
    
    print('---')
    
plt.plot(losses_train, label='train')
plt.plot(losses_test, label='test')
plt.savefig('models/losses.png')

net = net.cpu()
torch.save(net.state_dict(),'models/net_self_trained.pth')