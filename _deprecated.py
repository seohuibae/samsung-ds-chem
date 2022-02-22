'''
code for multi-heads + scaling 
'''
import os 
import argparse
from tqdm import tqdm 
import numpy as np 
import collections

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import DataLoader as DataLoaderText
from torch_geometric.data import DataLoader

from models.gcn import GCN
from models.gat import GAT
from models.gin import GIN 
from models.transformer import Transformer, SimpleTextClassificationModel
from dataset import set_dataset

from dataset import wrapper_for_collate_batch
from deepchem.feat.smiles_tokenizer import SmilesTokenizer, BasicSmilesTokenizer
from utils import dict2tsv, MultipleOptimizer, set_seed

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

parser = argparse.ArgumentParser(description='PyTorch Training')

parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--epochs', default=50, type=int)                   
parser.add_argument('--ft_epochs', default=20, type=int)
parser.add_argument('--lr', default=1e-2, type=float)
parser.add_argument('--ft_lr', default=1e-2, type=float)
parser.add_argument('--model', default='GCN', type=str, choices=['simple','transformer', 'GCN', 'GAT', 'GIN'])
parser.add_argument('--gpu', default=0, type=int) # 0,1,2,3,
parser.add_argument('--finetune_only', action='store_true' )
parser.add_argument('--k_fold', default=None, type=int) # 5 
parser.add_argument('--seed', default=2020, type=int)
args = parser.parse_args()

device = torch.device("cuda:"+args.gpu if torch.cuda.is_available() else "cpu")

def scaled_l2(pred, y, scale):
    return ((pred - y) ** 2 / scale).mean()

def build_optimizer(model, optim, lr):
    if optim=='adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optim == 'sparseadam':
        dense = []
        sparse = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            # TODO: Find a better way to check for sparse gradients.
            if 'embed' in name:
                sparse.append(param)
            else:
                dense.append(param)
        optimizer = MultipleOptimizer(
            [torch.optim.Adam(
                dense,
                lr=lr),
            torch.optim.SparseAdam(
                sparse,
                lr=lr)])
    else:
        raise NotImplementedError
    return optimizer

def main(args):
    
    set_seed(args.seed, use_cuda=True)

    for test_idx in range(4):
        
        vocab_rootdir='./vocab/'
        vocab_path = os.path.join(vocab_rootdir,'vocab.txt')
        tokenizer = SmilesTokenizer(vocab_path)
        
        ### TODO 5-fold 
        if args.model in ['simple', 'transformer']:
            train_dataset, test_sets = set_dataset(root='./polyinfo/raw', test_idx=test_idx, use_smiles_only=True, k_fold=args.k_fold) 
        else: 
            train_dataset, test_sets = set_dataset(root='./polyinfo/raw', test_idx=test_idx, k_fold=args.k_fold)

        if args.model in ['simple', 'transformer']:
            train_loader = DataLoaderText(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=wrapper_for_collate_batch(tokenizer))
        else:
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            
        model_dict = {
            'simple': SimpleTextClassificationModel(vocab_size=tokenizer.vocab_size, embed_dim=64, num_class=4),
            'transformer': Transformer(vocab_size=tokenizer.vocab_size, embed_dim=64, num_class=4, num_layers=6, nhead=2, d_model=64 ),
            'GCN':  GCN(num_input_features=9, num_layers=3, hidden_dim=64, out_dim=64),
            'GAT': GAT(num_input_features=9, num_layers=3, hidden_dim=64, out_dim=64, heads=8),
            'GIN': GIN(num_input_features=9, num_layers=3, hidden_dim=64, out_dim=64)
        }
    
        model = model_dict[args.model] 
    
        model = model.to(device)
        if args.model in ['simple', 'transformer']:
            optimizer = build_optimizer(model, 'sparseadam', args.lr) 
        else: 
            optimizer = build_optimizer(model, 'adam', args.lr)

        criterion = nn.MSELoss()
        
        if not os.path.exists(f'./multi-{args.model}'):
            os.mkdir(f'./multi-{args.model}')
        
        if not os.path.exists(f'./multi-{args.model}/{args.seed}'):
            os.mkdir(f'./multi-{args.model}/{args.seed}')
            
        # pre-train 
        print(args.finetune_only)
        if args.finetune_only:
            print('load pre-trained model')
            model.load_state_dict(torch.load(f'./multi-{args.model}/{args.seed}/pretrained-{test_idx}.pth.tar'))
        else: 
            for epoch in range(args.epochs):
                model.train()

                training_loss = []
                for batch in tqdm(train_loader):
                    if args.model in ['simple', 'transformer']:
                        cls, text, offsets = batch 
                        cls, text, offsets = cls.to(device), text.to(device), offsets.to(device)
                        bsz = len(cls)
                        pred = model(text, offsets)
                        _y = cls
    
                    else: 
                        bsz = batch.num_graphs
                        batch.to(device)  
                        pred = model(batch)        
                        _y = batch.y                           # pred: [bsz, 4]
                    
                    y  = torch.tensor([val[0] for val in _y]).float()
                    y_idx = torch.tensor([val[1] for val in _y])
                    y_mean  = torch.tensor([val[2] for val in _y]).float()
                    # y_std  = torch.tensor([val[3] for val in batch.y]).float()
                    if args.model == 'transformer':
                        pred = pred.view(-1, 4)
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
                dict2tsv(res, f'./multi-{args.model}/{args.seed}/logs_pretrain-{test_idx}.txt')
            torch.save(model.state_dict(), f'./multi-{args.model}/{args.seed}/pretrained-{test_idx}.pth.tar')
            
        ######
        # model = GCN(num_input_features=9, num_layers=3, hidden_dim=64, out_dim=64)
        # model = model.to(device)
        # model.load_state_dict(torch.load('./pretrained-{}.pth.tar'.format(test_idx)))
        total_test_iter = args.k_fold if args.k_fold is not None else 1 

        if args.model in ['simple', 'transformer']:
            optimizer = build_optimizer(model, 'sparseadam', args.ft_lr) 
        else: 
            optimizer = build_optimizer(model, 'adam', args.ft_lr)

        test_loss_sets, test_r2_sets, test_rmse_sets, test_mae_sets = [], [], [], []
        for k in range(total_test_iter):    
            print(f'{k}/{total_test_iter} folds')
            ft_dataset, test_dataset = test_sets[k]

            if args.model in ['simple', 'transformer']:
                ft_loader = DataLoaderText(ft_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=wrapper_for_collate_batch(tokenizer))
                test_loader = DataLoaderText(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=wrapper_for_collate_batch(tokenizer))
            else: 
                ft_loader = DataLoader(ft_dataset, batch_size=args.batch_size, shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

            # fine-tune
            for epoch in range(args.ft_epochs):
                model.train()
                training_loss = []
                for batch in tqdm(ft_loader):
                    if args.model in ['simple', 'transformer']:
                        cls, text, offsets = batch 
                        cls, text, offsets = cls.to(device), text.to(device), offsets.to(device)
                        bsz = len(cls)
                        pred = model(text, offsets)                    
                        _y = cls
                    else: 
                        bsz = batch.num_graphs
                        batch.to(device)  
                        pred = model(batch)        
                        _y = batch.y                           # pred: [bsz, 4]
                    
                    y  = torch.tensor([val[0] for val in _y]).float()
                    y_idx = torch.tensor([val[1] for val in _y])
                    y_mean  = torch.tensor([val[2] for val in _y]).float()

                    if args.model == 'transformer':
                        pred = pred.view(-1, 4)
                    
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
                dict2tsv(res, f'./multi-{args.model}/{args.seed}/logs_finetune-{test_idx}.txt')
            
            # test 
            model.eval()
            test_loss = []
            ys = []
            preds = []
            with torch.no_grad():
                for batch in tqdm(test_loader):
                    if args.model in ['simple', 'transformer']:
                        cls, text, offsets = batch 
                        cls, text, offsets = cls.to(device), text.to(device), offsets.to(device)
                        bsz = len(cls)
                        pred = model(text, offsets)                    
                        _y = cls

                    else: 
                        bsz = batch.num_graphs
                        batch.to(device)  
                        pred = model(batch)        
                        _y = batch.y                           # pred: [bsz, 4]
                    
                    y  = torch.tensor([val[0] for val in _y]).float()
                    y_idx = torch.tensor([val[1] for val in _y])

                    if args.model == 'transformer':
                        pred = pred.view(-1, 4)
                
                    pred = pred[torch.arange(bsz).view(1,-1), y_idx]    
                    pred = pred.squeeze(0)

                    loss = criterion(pred, y)
                    test_loss.append(loss.item())
                    ys.append(y.detach().numpy())
                    preds.append(pred.detach().numpy())
            ys = np.concatenate(ys)
            preds = np.concatenate(preds)

            test_loss = np.mean(test_loss)
            test_r2 = r2_score(ys, preds)
            test_rmse = np.sqrt(mean_squared_error(ys, preds)) 
            test_mae = mean_absolute_error(ys, preds)
            print('[test] loss:{:.5f} '.format(test_loss))
            print('[test] r2:{:.5f} '.format(test_r2))
            print('[test] rmse:{:.5f} '.format(test_rmse))
            print('[test] mae:{:.5f} '.format(test_mae))

            test_loss_sets.append(test_loss)
            test_r2_sets.append(test_r2)
            test_rmse_sets.append(test_rmse)
            test_mae_sets.append(test_mae)

        # sets 
        print('=== final ===')
        print('[test] loss:{:.5f} '.format(np.mean(test_loss_sets)))
        print('[test] r2:{:.5f} '.format(np.mean(test_r2_sets)))
        print('[test] rmse:{:.5f} '.format(np.mean(test_rmse_sets)))
        print('[test] mae:{:.5f} '.format(np.mean(test_mae_sets)))

        res = collections.OrderedDict()
        res['loss'] = np.mean(test_loss_sets)
        res['r2'] = np.mean(test_r2_sets)
        res['rmse'] = np.mean(test_rmse_sets)
        res['mae'] = np.mean(test_mae_sets) 

        dict2tsv(res, f'./multi-{args.model}/{args.seed}/logs_test-{test_idx}.txt')


if __name__ == '__main__':
    main(args)
