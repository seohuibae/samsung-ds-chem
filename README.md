# Samsung-DS-Chemical
## Dataset 
PolyInfo 

## Requirements
pytorch >= 1.8.0 \
pytorch geometric (https://github.com/rusty1s/pytorch_geometric) \
rdkit (https://github.com/rdkit/rdkit)

## Run
### single-task (baseline)
python main_train_single_tuning_gcn.py --num_layers 6 --hidden_dim 200 --dropout 0.1 --lr 1e-03 --l2reg 1e-02 --mlp_dropout 0.1 --mlp_dims 50 --model GCN --k_fold 5 --epochs 300 --gpu 0

### multi-task
python main_train_multiheads_tuning_gcn.py --num_layers 6 --hidden_dim 200 --dropout 0.1 --lr 5e-04 --ft_lr 5e-04 --l2reg 1e-02 --mlp_dropout 0.1 --mlp_dims 200 --model GCN --k_fold 5 --epochs 300 --ft_epochs 100 --gpu 0


