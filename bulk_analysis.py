import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats.mstats import winsorize
from scipy import stats
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from collections import Counter
import scanpy as sc
import anndata as an
import sys
import os
import gc

folder = 'data/'

lst = os.listdir(folder)
bulkfiles = [f[:-9] for f in lst if f.endswith('_bulk.tsv')]
print(len(bulkfiles),'files')
print(bulkfiles)

merged_adata = None
for i,f in enumerate(bulkfiles):
    bulk_df = pd.read_csv(folder+f+'_bulk.tsv',sep='\t',index_col=0).T
    cellnumbers_df = pd.read_csv(folder+f+'_numberofcells.txt',sep='\t',index_col=0)
    valid_conditions = list(cellnumbers_df[cellnumbers_df['NUMBER_OF_CELLS']>100].index)
    print(i+1,f,len(bulk_df),len(valid_conditions))
    bulk_df = bulk_df.loc[valid_conditions]
    bulk_df.index = [f+'_'+vc for vc in valid_conditions]
    if merged_adata is None:
        merged_adata = an.AnnData(X=bulk_df)
    else:
        temp_adata = an.AnnData(X=bulk_df)
        merged_adata = sc.concat([merged_adata,temp_adata])
    #if i>10:
    #    break
    del bulk_df
    gc.collect()

merged_adata.obs['plate'] = [s.split('_')[0] for s in merged_adata.obs_names]
merged_adata.obs['cellline'] = [s.split('_')[1]+'_'+s.split('_')[2] for s in merged_adata.obs_names]
merged_adata.obs['drug'] = [s.split('_')[3] for s in merged_adata.obs_names]
merged_adata.obs['drugconc'] = [s.split('_')[3]+s.split('_')[4] for s in merged_adata.obs_names]

merged_adata.write_h5ad(folder+'tahoe_bulk.h5ad',compression='gzip')

sc.pp.normalize_total(merged_adata, target_sum=1000000)
top_variable_genes = 20000
vars = np.var(merged_adata.X,axis=0)
inds = np.flip(np.argsort(vars))
ind_genes = inds[0:top_variable_genes]
if 0 in vars[ind_genes]:
    ind_first_zero = np.argwhere(vars[ind_genes]==0)[0][0]
    ind_genes = ind_genes[0:ind_first_zero]
adata_bulk = merged_adata[:,ind_genes]
sc.pp.log1p(adata_bulk)
sc.tl.pca(adata_bulk)

adata_bulk.obs['plate'] = merged_adata.obs['plate']
adata_bulk.obs['cellline'] = merged_adata.obs['cellline']
adata_bulk.obs['drug'] = merged_adata.obs['drug']
adata_bulk.obs['drugconc'] = merged_adata.obs['drugconc']

adata_bulk.write_h5ad(folder+f'tahoe_bulk_processed{top_variable_genes}.h5ad',compression='gzip')
