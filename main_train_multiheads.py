'''
code for multi-heads + scaling 
'''

import argparse
from tqdm import tqdm 
import numpy as np 
import collections

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch_geometric.data import DataLoader

from models.gcn import GCN
from dataset import set_dataset
from utils import dict2tsv

parser = argparse.ArgumentParser(description='PyTorch Training')

parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--epochs', default=50, type=int)                   
parser.add_argument('--ft_epochs', default=20, type=int)
parser.add_argument('--lr', default=1e-2, type=float)
parser.add_argument('--ft_lr', default=1e-2, type=float)

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def scaled_l2(pred, y, scale):
    return ((pred - y) ** 2 / scale).mean()


def main(args):
    for test_idx in range(4):
        train_dataset, ft_dataset, test_dataset = set_dataset(root='./polyinfo/raw', test_idx=test_idx)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        ft_loader = DataLoader(ft_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
        model = GCN(num_input_features=9, num_layers=3, hidden_dim=64, out_dim=64)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  # TODO: tuning 

        criterion = nn.MSELoss()

        # pre-train 
        for epoch in range(args.epochs):
            model.train()

            training_loss = []
            for batch in tqdm(train_loader):
                bsz = batch.num_graphs
                batch.to(device)  
                pred = model(batch)                                   # pred: [bsz, 4]
                
                y  = torch.tensor([val[0] for val in batch.y]).float()
                y_idx = torch.tensor([val[1] for val in batch.y])
                y_mean  = torch.tensor([val[2] for val in batch.y]).float()
                # y_std  = torch.tensor([val[3] for val in batch.y]).float()

                pred = pred[torch.arange(bsz).view(1,-1), y_idx]    
                pred = pred.squeeze(0)

                loss = scaled_l2(pred, y, y_mean)
                # loss = criterion(pred, y)
                training_loss.append(loss.item())
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            print('[pre-train {}-epoch] train loss:{:.5f} '.format(epoch+1, np.mean(training_loss)))

            res = collections.OrderedDict()
            res['epoch'] = epoch
            res['loss'] = np.mean(training_loss)
            # print(res)
            dict2tsv(res, './logs_pretrain-{}.txt'.format(test_idx))
        torch.save(model.state_dict(), './pretrained-{}.pth.tar'.format(test_idx))
        
        ######
        # model = GCN(num_input_features=9, num_layers=3, hidden_dim=64, out_dim=64)
        # model = model.to(device)
        # model.load_state_dict(torch.load('./pretrained-{}.pth.tar'.format(test_idx)))

        optimizer = torch.optim.Adam(model.parameters(), lr=args.ft_lr)  
        # fine-tune
        for epoch in range(args.ft_epochs):
            model.train()
            training_loss = []
            for batch in tqdm(ft_loader):
                bsz = batch.num_graphs
                batch.to(device)  
                pred = model(batch)                                   

                y  = torch.tensor([val[0] for val in batch.y]).float()
                y_idx = torch.tensor([val[1] for val in batch.y])
                y_mean  = torch.tensor([val[2] for val in batch.y]).float()
                
                pred = pred[torch.arange(bsz).view(1,-1), y_idx]    
                pred = pred.squeeze(0)

                # loss = criterion(pred, y)
                loss = scaled_l2(pred, y, y_mean)
                training_loss.append(loss.item())
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print('[fine-tune {}-epoch] train loss:{:.5f} '.format(epoch+1, np.mean(training_loss)))
            
            res = collections.OrderedDict()
            res['epoch'] = epoch
            res['loss'] = np.mean(training_loss)
            dict2tsv(res, './logs_finetune-{}.txt'.format(test_idx))
        
        # test 
        model.eval()
        test_loss = []
        with torch.no_grad():
            for batch in tqdm(test_loader):
                bsz = batch.num_graphs
                batch.to(device)  
                pred = model(batch)                                   
                
                y  = torch.tensor([val[0] for val in batch.y]).float()
                y_idx = torch.tensor([val[1] for val in batch.y])
            
                pred = pred[torch.arange(bsz).view(1,-1), y_idx]    
                pred = pred.squeeze(0)

                loss = criterion(pred, y)
                test_loss.append(loss.item())
        print('[test] loss:{:.5f} '.format(np.mean(test_loss)))

        res = collections.OrderedDict()
        res['loss'] = np.mean(test_loss)
        dict2tsv(res, './logs_test-{}.txt'.format(test_idx))


if __name__ == '__main__':
    main(args)