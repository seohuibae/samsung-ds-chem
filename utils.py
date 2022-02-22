import os  
import pandas as pd 
import matplotlib.pyplot as plt

import random
import numpy as np 
import torch 

# https://github.com/sehkmg/tsvprint/blob/master/utils.py
def dict2tsv(res, file_name):
    if not os.path.exists(file_name):
        with open(file_name, 'a') as f:
            f.write('\t'.join(list(res.keys())))
            f.write('\n')

    with open(file_name, 'a') as f:
        f.write('\t'.join([str(r) for r in list(res.values())]))
        f.write('\n')
def plot(file_name='result.txt'):
    result = pd.read_csv(file_name, delimiter='\t').values
    plt.figure()
    plt.plot(result[:, 0], result[:, 1])
    plt.savefig('result.png')

def set_seed(seed, use_cuda=True):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False 
        torch.backends.cudnn.deterministic = True 


class MultipleOptimizer(object):
    """ Implement multiple optimizers needed for sparse adam """

    def __init__(self, op):
        """ ? """
        self.optimizers = op

    @property
    def param_groups(self):
        param_groups = []
        for optimizer in self.optimizers:
            param_groups.extend(optimizer.param_groups)
        return param_groups

    def zero_grad(self):
        """ ? """
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        """ ? """
        for op in self.optimizers:
            op.step()

    @property
    def state(self):
        """ ? """
        return {k: v for op in self.optimizers for k, v in op.state.items()}

    def state_dict(self):
        """ ? """
        return [op.state_dict() for op in self.optimizers]

    def load_state_dict(self, state_dicts):
        """ ? """
        assert len(state_dicts) == len(self.optimizers)
        for i in range(len(state_dicts)):
            self.optimizers[i].load_state_dict(state_dicts[i])

if __name__ == '__main__':
    # plot()
    name_list = ['d', 'm', 'tg', 'tm']
    plt.figure()
    for i in range(4):
        file_name = 'logs_pretrain-{}.txt'.format(i)
        result = pd.read_csv(file_name, delimiter='\t').values
        plt.plot(result[:, 0], result[:, 1],label=name_list[i])
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('training loss')
    plt.savefig('result.png')
