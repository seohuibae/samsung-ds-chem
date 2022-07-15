import pubchempy as pcp 
import pandas as pd 
import numpy as np 
# c = pcp.Compound.from_cid
def parse_name(input):
    tmp = input.strip().split('poly')
    if tmp[0]!="":
        name = "".join(tmp)
    else: 
        name = tmp[1]
        if name[0] in ['(','{','[']:
            name = name[1:-1]
    return name 
    
df = pd.read_csv('polyinfo/raw/K-raw.csv', encoding='latin-1')
names = df['name'].tolist()
name_df=[]
smiles_df=[]
value_df=[]
for i,row in df.iterrows(): 
    print(i)
    name = parse_name(row['name'])
    print(name)
    compounds = pcp.get_compounds(name,'name')
    isomeric_smiles = ""
    print(compounds)
    for compound in compounds:
        isomeric_smiles = compound.isomeric_smiles
        break 
    if isomeric_smiles !="":
        isomeric_smiles="*"+isomeric_smiles+"*"
        if np.float(row['value1'])==np.float(row['value2']):
            name_df.append(row['name'])
            smiles_df.append(isomeric_smiles)
            value_df.append(np.mean([np.float(row['value1']),np.float(row['value2'])]))
    
smiles_df = pd.DataFrame(smiles_df)
value_df = pd.DataFrame(value_df)
df_merged = pd.concat([smiles_df, value_df], axis=1, ignore_index=True) 
# df_merged.to_csv('polyinfo/raw/K_v0.0.csv', header=None, index=None)
df_merged.to_csv('polyinfo/raw/K_v1.0.csv', header=None, index=None)

 

