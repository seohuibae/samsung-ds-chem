import os 
import itertools

# for gat
hparams_space = {'num_layers':[3,4], 'hidden_dim':[64, 128],'dropout': [0.1, 0.2]}
keys, values = zip(*hparams_space.items())
hparams_permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]


if __name__ == '__main__':
    model = 'gin'
    seed = 2024
    idx = 0

    f = open(f'{model}_{seed}.sh' ,'w')
    for hparams in hparams_permutations:
        model_path = []
        for k,v in hparams.items():
            model_path.append(str(k)+str(v))
        model_path = '-'.join(model_path)
        # print(hparams)
        string = f"python main_train_multiheads_tuning_{model}.py --tuning --k_fold 5 --seed {seed} --num_layers {hparams['num_layers']} --hidden_dim {hparams['hidden_dim']} --dropout {hparams['dropout']} --model_path {model_path} --gpu {(idx%4)} &\n"
        f.write(string)
        

        idx+=1
    f.close()