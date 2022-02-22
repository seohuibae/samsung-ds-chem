'''
[Reference]
    https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html
    https://github.com/deepfindr/gnn-project/blob/main/dataset.py
'''

import os
import numpy as np 
import pandas as pd
from tqdm import tqdm
from itertools import chain 
import random

import torch
import torch_geometric
from torch_geometric.data import Dataset, Data

from rdkit import Chem

import io 
import os
import pickle 

# from torchtext.vocab import build_vocab_from_iterator
from collections import Counter, OrderedDict 

### TODO datset for cl 

def set_dataset_single_task(root, test_idx, use_smiles_only=False, k_fold=None):
    files = ['Density_v0.0.csv', 'Modulus_v0.0.csv', 'Tg-clan_v0.0.csv', 'Tm-clan_v0.0.csv']
    train_files = ['Density_v0.0.csv', 'Modulus_v0.0.csv', 'Tg-clan_v0.0.csv', 'Tm-clan_v0.0.csv']
    
    test_file = train_files.pop(test_idx)
    train_files = test_file 

    # dataset for pre-train # no pretrain 
    train_data = []
    # for train_file in train_files:
    #     file_path = os.path.join(root, train_file)
    #     pd_data = pd.read_csv(file_path).values
    #     train_idx  = files.index(train_file)
        
    #     if train_file in ['Density_v0.0.csv', 'Modulus_v0.0.csv']:
    #         smiles_idx = 1 
    #         label_idx = 2
    #     else:
    #         smiles_idx = 0
    #         label_idx = 1 
           
    #     tmp = []
    #     for mol in tqdm(pd_data, total=pd_data.shape[0]):
    #         tmp.append(mol[label_idx])
    #     tmp = np.array(tmp)
            
    #     for mol in tqdm(pd_data, total=pd_data.shape[0]):
    #         data = parse_mol(mol, test_idx, smiles_idx, label_idx, tmp)
    #         if use_smiles_only:
    #             data = (data.y, data.smiles) 
    #         train_data.append(data)
            
    # eval dataset(fine-tune, test)
    eval_file_path = os.path.join(root, test_file)
    pd_data = pd.read_csv(eval_file_path).values
    num_eval_data = len(pd_data)
    
    # split the eval dataset
    if k_fold is None:
        ft_idx = list(np.random.choice(num_eval_data, num_eval_data // 5, replace=False))
        ft_idx_sets = [ft_idx]
        # test_idx = list(set(list(range(num_eval_data))) - set(ft_idx))
        ft_data = []; test_data = []

    else: # if int
        print(k_fold)
        eval_data_list = [i for i in range(num_eval_data)]
        random.shuffle(eval_data_list)
        num_sample_per_fold = num_eval_data//k_fold
        ft_idx_sets = []
        for k in range(k_fold):
            ft_idx_sets.append(eval_data_list[num_sample_per_fold*k:num_sample_per_fold*(k+1)])
    
    total_test_iter = k_fold if k_fold is not None else 1 

    test_sets = []
    for k in range(total_test_iter):
        ft_idx = ft_idx_sets[k]

        ft_data = []; test_data = []
        if test_file in ['Density_v0.0.csv', 'Modulus_v0.0.csv']:
            smiles_idx = 1 
            label_idx = 2
        else:
            smiles_idx = 0
            label_idx = 1 

        tmp = []
        for i, mol in enumerate(pd_data):    
            if i in ft_idx:
                tmp.append(mol[label_idx])
        tmp = np.array(tmp)
        for i, mol in tqdm(enumerate(pd_data), total=pd_data.shape[0]):
            data = parse_mol(mol, test_idx, smiles_idx, label_idx, tmp)
            if use_smiles_only:
                data = (data.y, data.smiles) 
            # fine-tune dataset 
            if i in ft_idx:
                ft_data.append(data)
            # test dataset 
            else:
                test_data.append(data)
        test_sets.append((ft_data, test_data))
        # test_sets.append((ft_data, test_data))
    # return train_data, ft_data, test_data 
    return train_data, test_sets

def set_dataset(root, test_idx, use_smiles_only=False, k_fold=None):
    files = ['Density_v0.0.csv', 'Modulus_v0.0.csv', 'Tg-clan_v0.0.csv', 'Tm-clan_v0.0.csv']
    train_files = ['Density_v0.0.csv', 'Modulus_v0.0.csv', 'Tg-clan_v0.0.csv', 'Tm-clan_v0.0.csv']
    
    test_file = train_files.pop(test_idx)

    # dataset for pre-train
    train_data = []
    for train_file in train_files:
        file_path = os.path.join(root, train_file)
        pd_data = pd.read_csv(file_path).values
        train_idx  = files.index(train_file)
        
        if train_file in ['Density_v0.0.csv', 'Modulus_v0.0.csv']:
            smiles_idx = 1 
            label_idx = 2
        else:
            smiles_idx = 0
            label_idx = 1 
           
        tmp = []
        for mol in tqdm(pd_data, total=pd_data.shape[0]):
            tmp.append(mol[label_idx])
        tmp = np.array(tmp)
            
        for mol in tqdm(pd_data, total=pd_data.shape[0]):
            data = parse_mol(mol, test_idx, smiles_idx, label_idx, tmp)
            if use_smiles_only:
                data = (data.y, data.smiles) 
            train_data.append(data)
            
    # eval dataset(fine-tune, test)
    eval_file_path = os.path.join(root, test_file)
    pd_data = pd.read_csv(eval_file_path).values
    num_eval_data = len(pd_data)
    
    # split the eval dataset
    
    if k_fold is None:
        ft_idx = list(np.random.choice(num_eval_data, num_eval_data // 10, replace=False))
        ft_idx_sets = [ft_idx]
        # test_idx = list(set(list(range(num_eval_data))) - set(ft_idx))
        ft_data = []; test_data = []

    else: # if int
        print(k_fold)
        eval_data_list = [i for i in range(num_eval_data)]
        random.shuffle(eval_data_list)
        num_sample_per_fold = num_eval_data//k_fold
        ft_idx_sets = []
        for k in range(k_fold):
            ft_idx_sets.append(eval_data_list[num_sample_per_fold*k:num_sample_per_fold*(k+1)])
    
    total_test_iter = k_fold if k_fold is not None else 1 

    test_sets = []
    for k in range(total_test_iter):
        ft_idx = ft_idx_sets[k]

        ft_data = []; test_data = []
        if test_file in ['Density_v0.0.csv', 'Modulus_v0.0.csv']:
            smiles_idx = 1 
            label_idx = 2
        else:
            smiles_idx = 0
            label_idx = 1 

        tmp = []
        for i, mol in enumerate(pd_data):    
            if i in ft_idx:
                tmp.append(mol[label_idx])
        tmp = np.array(tmp)
        for i, mol in tqdm(enumerate(pd_data), total=pd_data.shape[0]):
            data = parse_mol(mol, test_idx, smiles_idx, label_idx, tmp)
            if use_smiles_only:
                data = (data.y, data.smiles) 
            # fine-tune dataset 
            if i in ft_idx:
                ft_data.append(data)
            # test dataset 
            else:
                test_data.append(data)
        test_sets.append((ft_data, test_data))
    # return train_data, ft_data, test_data 
    return train_data, test_sets
    
def wrapper_for_collate_batch(tokenizer):
    
    def collate_batch(batch):
        text_pipeline = lambda x: tokenizer.encode(x)
        label_list, text_list, offsets = [], [], [0]
        for (_label, _text) in batch:
            # label_list.append(label_pipeline(_label))
            label_list.append(torch.tensor(_label))
            processed_text = torch.tensor(text_pipeline(_text.strip().strip('*')), dtype=torch.int64)
            text_list.append(processed_text)
            offsets.append(processed_text.size(0))
        label_list = torch.cat(label_list)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        text_list = torch.cat(text_list)

        return label_list, text_list, offsets
    return collate_batch


def parse_mol(mol, train_idx, smiles_idx, label_idx, tmp=[]):
    smiles = mol[smiles_idx]
    if tmp == []:
        label = [mol[label_idx], train_idx, 0, 0]
    else:
        label = [mol[label_idx], train_idx, tmp.mean(), tmp.std()]
        # label = [(mol[label_idx] - tmp.min()) / (tmp.max() - tmp.min()), train_idx]
    
    mol_obj = Chem.MolFromSmiles(smiles)
    # Get node features
    node_feats = get_node_features(mol_obj)
    # Get edge features
    edge_feats = get_edge_features(mol_obj)
    # Get adjacency info
    edge_index = get_adjacency_info(mol_obj)
    
    # Get labels info
    label = get_labels(label)

    # Create data object
    data = Data(x=node_feats, 
                edge_index=edge_index,
                edge_attr=edge_feats,
                y=label,
                smiles=smiles
            ) 
    return data 

def get_node_features(mol):
    """ 
    This will return a matrix / 2d array of the shape
    [Number of Nodes, Node Feature size]
    """
    all_node_feats = []

    for atom in mol.GetAtoms():
        node_feats = []
        # Feature 1: Atomic number        
        node_feats.append(atom.GetAtomicNum())
        # Feature 2: Atom degree
        node_feats.append(atom.GetDegree())
        # Feature 3: Formal charge
        node_feats.append(atom.GetFormalCharge())
        # Feature 4: Hybridization
        node_feats.append(atom.GetHybridization())
        # Feature 5: Aromaticity
        node_feats.append(atom.GetIsAromatic())
        # Feature 6: Total Num Hs
        node_feats.append(atom.GetTotalNumHs())
        # Feature 7: Radical Electrons
        node_feats.append(atom.GetNumRadicalElectrons())
        # Feature 8: In Ring
        node_feats.append(atom.IsInRing())
        # Feature 9: Chirality
        node_feats.append(atom.GetChiralTag())

        # Append node features to matrix
        all_node_feats.append(node_feats)

    all_node_feats = np.asarray(all_node_feats)
    return torch.tensor(all_node_feats, dtype=torch.float)

def get_edge_features(mol):
    """ 
    This will return a matrix / 2d array of the shape
    [Number of edges, Edge Feature size]
    """
    all_edge_feats = []

    for bond in mol.GetBonds():
        edge_feats = []
        # Feature 1: Bond type (as double)
        edge_feats.append(bond.GetBondTypeAsDouble())
        # Feature 2: Rings
        edge_feats.append(bond.IsInRing())
        # Append node features to matrix (twice, per direction)
        all_edge_feats += [edge_feats, edge_feats]

    all_edge_feats = np.asarray(all_edge_feats)
    return torch.tensor(all_edge_feats, dtype=torch.float)

def get_adjacency_info(mol):
    """
    We could also use rdmolops.GetAdjacencyMatrix(mol)
    but we want to be sure that the order of the indices
    matches the order of the edge features
    """
    edge_indices = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices += [[i, j], [j, i]]

    edge_indices = torch.tensor(edge_indices)
    edge_indices = edge_indices.t().to(torch.long).view(2, -1)
    return edge_indices

def get_labels(label):
    label = np.asarray([label])
    return torch.tensor(label, dtype=torch.float64)


# def encode_tokens(test_idx, rootdir, vocab_rootdir='./vocab/'):
#     train_files = ['Density_v0.0.csv', 'Modulus_v0.0.csv', 'Tg-clan_v0.0.csv', 'Tm-clan_v0.0.csv']
#     test_file = train_files.pop(test_idx)
    
#     vocab_path = os.path.join(vocab_rootdir,'vocab.txt')
#     tokenizer = SmilesTokenizer(vocab_path)

#     for filename in train_files:
#         filepath = rootdir+filename
#         smiles_idx = 0
#         if 'Density' in filename or 'Modulus' in filename: 
#             smiles_idx = 1

#         with io.open(filepath, encoding='utf-8') as f:
#             for line in f: 
#                 line = line.split(',')[smiles_idx] # smiles column
#                 yield tokenizer.encode(line)
