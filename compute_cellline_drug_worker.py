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
sys.path.append('./code/scycle/')
import scycle as cc
import os
import pertpy as pt
import gc
import pickle
import seaborn as sns
import traceback

def smooth_adata_by_pooling(adata,X_embed,n_neighbours=10):
    adata_pooled = adata.copy()
    nbrs = NearestNeighbors(n_neighbors=n_neighbours).fit(X_embed)
    distances, indices = nbrs.kneighbors(X_embed)    
    adata_pooled.X = smooth_matrix_by_pooling(get_nd_array(adata.X),indices)
    if 'matrix' in adata.layers:
        adata_pooled.layers['matrix'] = smooth_matrix_by_pooling(get_nd_array(adata.layers['matrix']),indices)
    if 'spliced' in adata.layers:
        adata_pooled.layers['spliced'] = smooth_matrix_by_pooling(get_nd_array(adata.layers['spliced']),indices)
    if 'unspliced' in adata.layers:
        adata_pooled.layers['unspliced'] = smooth_matrix_by_pooling(get_nd_array(adata.layers['unspliced']),indices)
    return adata_pooled

def smooth_matrix_by_pooling(matrix,indices):
    matrix_pooled = matrix.copy()
    for i in range(len(indices)):
        matrix_pooled[i,:] = np.mean(matrix[indices[i],:],axis=0)
    return matrix_pooled

def get_nd_array(arr):
    x = None
    if str(type(arr)):
        x = arr
    else:
        x = arr.toarray()
    return x

    
def preprocessing_without_pooling(adata,
                                  Normalize_Totals=True,
                                  top_variable_genes=10000,
                                  Already_Log_Transformed=False,
                                  number_of_pcs=30):
    if Normalize_Totals:
        sc.pp.normalize_total(adata, target_sum=10000)
    if top_variable_genes>0:
        #sc.pp.highly_variable_genes(adata,n_top_genes=top_variable_genes,n_bins=20)
        #ind_genes = np.where(adata.var['highly_variable'])[0]
        vars = np.var(adata.X,axis=0)
        inds = np.flip(np.argsort(vars))
        ind_genes = inds[0:top_variable_genes]
        if 0 in vars[ind_genes]:
            ind_first_zero = np.argwhere(vars[ind_genes]==0)[0][0]
            ind_genes = ind_genes[0:ind_first_zero]
        #print(vars[ind_genes])
        adata = adata[:,ind_genes]
    if not Already_Log_Transformed:
        sc.pp.log1p(adata)
    sc.tl.pca(adata,n_comps=number_of_pcs)
    return adata

# pooling procedure
def pooling_procedure(adata,adata_orig,
                      n_neighbours_for_pooling=10,
                      number_of_pcs=30):
    if n_neighbours_for_pooling>0:    
        adata_work = adata_orig.copy()
        preprocessing_without_pooling(adata)
        sc.tl.pca(adata,n_comps=number_of_pcs)
        X_pca = adata.obsm['X_pca']
        adata = smooth_adata_by_pooling(adata_work,X_pca,n_neighbours=n_neighbours_for_pooling)
    return adata

def ismember(A, B):
    dct = {}
    for s,i in enumerate(B):
        dct[i] = s
    return [ dct[a] for a in A ]

def load_signature_file(file):
    sigs = {}
    with open(file,'r',encoding="utf8",errors='ignore') as fin:
        line = fin.readline().strip('\n').strip(' ')
        while line:
            parts = line.split('\t')
            lst = parts[2:]
            lst = [s.split('[')[0] for s in lst if not s=='']
            sigs[parts[0]] = lst
            line = fin.readline().strip('\n').strip(' ')
    return sigs

def load_weighted_signature_file(file):
    sigs = {}
    with open(file,'r',encoding="utf8",errors='ignore') as fin:
        line = fin.readline().strip('\n').strip(' ')
        while line:
            parts = line.split('\t')
            lst = parts[2:]
            lst1 = [s.split('[')[0] for s in lst if not s=='']
            weights = [float(s.split('[')[1].split(']')[0]) for s in lst if not s=='']
            #print(lst1,weights)
            sigs[parts[0]] = (lst1,weights)
            #sigs[parts[0]+'_W'] = weights
            line = fin.readline().strip('\n').strip(' ')
    return sigs

def calc_scores(anndata,signature_dict):
    matrix = anndata.to_df().to_numpy()
    scores_dic = {}
    for key in signature_dict:
        names = np.array(signature_dict[key])
        inds = np.where(np.isin(anndata.var_names,names))[0]
        matrix_sel = matrix[:,inds]
        scores = np.mean(matrix_sel,axis=1)
        scores_dic[key] = scores
        anndata.obs[key] = scores
    return scores_dic

def calc_weighted_scores(anndata,signature_dict):
    matrix = anndata.to_df().to_numpy()
    scores_dic = {}
    for key in signature_dict:
        names_weights = signature_dict[key]
        names = names_weights[0]
        weights = np.array(names_weights[1])
        inds = np.where(np.isin(anndata.var_names,names))[0]
        sel_names = anndata.var_names[inds]
        ind_in_names = ismember(sel_names,names)
        names1 = np.array(names)[ind_in_names]
        weights1 = np.array(weights)[ind_in_names]
        inds = ismember(names1,anndata.var_names)
        matrix_sel = matrix[:,inds]
        gene_means = np.mean(matrix_sel,axis=0)
        meanmat = np.outer(np.ones(matrix_sel.shape[0]),gene_means)
        matrix_sel = matrix_sel-meanmat
        scores = np.matmul(matrix_sel,weights1)
        scores_dic[key] = scores
    return scores_dic


def calc_histone_score(adata2k):
    histone_names1 = np.argwhere(adata2k.var_names.str.startswith('H1'))
    histone_names2 = np.argwhere(adata2k.var_names.str.startswith('H2'))
    histone_names3 = np.argwhere(adata2k.var_names.str.startswith('H3'))
    histone_names4 = np.argwhere(adata2k.var_names.str.startswith('H4'))
    histone_names5 = np.argwhere(adata2k.var_names.str.startswith('HIST'))
    histone_names = np.union1d(np.union1d(histone_names1,histone_names2),np.union1d(histone_names3,histone_names4))
    histone_names = np.union1d(histone_names,histone_names5)
    histone_names = adata2k.var_names[histone_names]
    print('Found histone genes:',*histone_names)
    inds_histones = np.where(np.isin(adata2k.var_names,histone_names))[0]
    matrix = adata2k.to_df().to_numpy()
    matrix_sel = matrix[:,inds_histones]
    scores = np.mean(matrix_sel,axis=1)
    return scores

def _compute_ica(adata,thr=2.0):
    adata.uns["scycle"] = {}
    cc.tl.dimensionality_reduction(adata,method='ica',find_cc_comp_thr=thr)

    idx_g1s = adata.uns['scycle']['find_cc_components']['indices']['G1-S']
    adata.uns['S-phase_genes'] = list(adata.var_names[adata.uns['dimRed'].S_[idx_g1s,:]>3])
    idx_g2m = adata.uns['scycle']['find_cc_components']['indices']['G2-M']
    adata.uns['G2-M_genes'] = list(adata.var_names[adata.uns['dimRed'].S_[idx_g2m,:]>3])
    #idx_g2m_inh = adata.uns['scycle']['find_cc_components']['indices']['G2-M-']
    #adata.uns['G2-M_INH_genes'] = list(adata.var_names[adata.uns['dimRed'].S_[idx_g2m_inh,:]>3])
    #idx_histone = adata.uns['scycle']['find_cc_components']['indices']['Histone']
    #adata.uns['Histone_IC_genes'] = list(adata.var_names[adata.uns['dimRed'].S_[idx_histone,:]>3])
    signature_dict = {'S-phase':adata.uns['S-phase_genes'],
                      'G2-M':adata.uns['G2-M_genes'],
                      #'G2-M-':adata.uns['G2-M_INH_genes'],
                      #'Histone_IC':adata.uns['Histone_IC_genes']
                      }  
    sc.pp.highly_variable_genes(adata,n_top_genes=2001,n_bins=20)
    ind_genes2k = np.where(adata.var['highly_variable'])[0]
    adata2k = adata[:,ind_genes2k]
    scores_dic = calc_scores(adata2k,signature_dict)
    for score in scores_dic:
        adata.obs[score] = scores_dic[score]
    adata.varm['P_dimRed'] = adata.uns['dimRed'].S_.T
    adata.uns['dimRed'] = None

def copy_celldrug_dataset(cell_line, drug, folder):
    code = os.system(f'gsutil cp gs://tahoe100m_bycelllines/panel_cell_drug/*"{cell_line}_{drug}_"*.h5ad {folder}')
    adata_cellline = None
    if code==0:
        files = os.listdir(folder)
        h5s = []
        plates = []
        for i,f in enumerate(files):
            if (cell_line+'_'+drug+'_' in f)&(f.startswith('plate')):
                if not f.startswith('plate14'):
                    print(i+1,f)
                    adata = sc.read_h5ad(folder+f)
                    h5s.append(adata)
                    plate = f.split('_')[0]
                    plates.append(plate)
        for p in plates:
            os.system(f'gsutil cp gs://tahoe100m_bycelllines/panel_cell_drug/{p}_{cell_line}_DMSO_TF_0.0.h5ad {folder}')
            adata = sc.read_h5ad(folder+f'{p}_{cell_line}_DMSO_TF_0.0.h5ad')    
            h5s.append(adata)
        adata_cellline = sc.concat(h5s)
        for h in h5s:
            del h       
    else:
        print(f'ERROR: Could not download the files for {cell_line}, {drug}')
    return adata_cellline

def cellline_drug_file_preprocessing(adata_cellline,
    top_variable_genes = 10000, # if negative then no selection of genes
    Normalize_Totals = True,
    Already_Log_Transformed = False,
    n_neighbours_for_pooling = 5,
    number_of_pcs = 30):
    
    print('PREPROCESSING PARAMETERS:')
    print('Already_Log_Transformed=',Already_Log_Transformed)
    print('Normalize_Totals=',Normalize_Totals)
    print('number_of_pcs=',number_of_pcs)
    print('n_neighbours_for_pooling=',n_neighbours_for_pooling)
    print('top_variable_genes=',top_variable_genes)
    
    if n_neighbours_for_pooling>0:
        adatat1 = pooling_procedure(adata_cellline,adata_cellline,
                                   number_of_pcs=number_of_pcs,
                                   n_neighbours_for_pooling=n_neighbours_for_pooling)
    else:
        adatat1 = adata_cellline
    adatat1 = preprocessing_without_pooling(adatat1,
                                      Normalize_Totals=True,
                                      top_variable_genes=10000,
                                      Already_Log_Transformed=False,
                                      number_of_pcs=30)
    return adatat1

def balance_control_measurement(adata_cellline):
    conds = np.unique(adata_cellline.obs['drugname_drugconc'])
    n_control = len(adata_cellline[adata_cellline.obs['drugname_drugconc']=="[('DMSO_TF', 0.0, 'uM')]"])
    print(f'{n_control=}')
    n_conds = []
    for cond in conds:
        if not cond=="[('DMSO_TF', 0.0, 'uM')]":
            n_cond = len(adata_cellline[adata_cellline.obs['drugname_drugconc']==cond])
            n_conds.append(n_cond)
    print(f'{n_conds=}')
    n_subs = np.max(np.array(n_conds))
    adata_control = adata_cellline[adata_cellline.obs['drugname_drugconc']=="[('DMSO_TF', 0.0, 'uM')]"]
    adata_control = sc.pp.sample(adata_control,fraction=float(n_subs)/float(n_control),copy=True)
    adatas = [adata_control]
    for cond in conds:
            if not cond=="[('DMSO_TF', 0.0, 'uM')]":
                adata_cond = adata_cellline[adata_cellline.obs['drugname_drugconc']==cond]
                adatas.append(adata_cond)
    adata_cellline = sc.concat(adatas)
    for a in adatas:
        del a
    del adata_control
    gc.collect()
    return adata_cellline

def get_cellcycle_space(adata_cellline):
    cc_genes = list(set(cc.data.cc_genes))
    cc_genes = [g for g in cc_genes if g in adata_cellline.var_names]
    adata_cellline = adata_cellline[:,cc_genes]
    return adata_cellline

def compute_distance_sameplate_control(adatat1,distance_type='edistance',plotheatmap=False,plotdistances=False):
    distance = pt.tl.Distance(distance_type, obsm_key="X_pca")
    plates = np.unique(adatat1.obs['plate'])
    dss_dict = {}
    for p in plates:
        print(p)
        dss = compute_distance(adatat1[adatat1.obs['plate']==p,:], 
                               distance_type=distance_type,
                               plotheatmap=plotheatmap,
                              plotdistances=plotdistances)
        for k in dss:
            if k in dss_dict:
                dss_dict[k] = np.min([dss[k],dss_dict[k]])
            else:
                dss_dict[k] = dss[k]    
    return dss_dict

def check_if_gsfile_exists(url):
    result = subprocess.run(['gsutil','ls',url], capture_output=True, text=True)
    out = result.stdout[:-1]
    return out==url

import ast
def compute_distance(adatat1,distance_type='edistance',plotheatmap=False,plotdistances=False):
    distance = pt.tl.Distance(distance_type, obsm_key="X_pca")
    df = distance.pairwise(adatat1, groupby='drugname_drugconc')
    if plotheatmap:
        sns.clustermap(df, robust=True, figsize=(10, 10))
        display(df)
    doses = []
    dists = []
    for c in df.columns:
        dose = ast.literal_eval(c)[0][1]
        #doses.append(-5.0 if dose==0 else np.log10(dose))
        doses.append(dose)
        dist = df.loc["[('DMSO_TF', 0.0, 'uM')]"][c].item()
        dists.append(dist)
    if plotdistances:
        df1 = pd.DataFrame({'logdose':doses,'edistance':dists})
        sns.scatterplot(df1,x='logdose',y='edistance',s=200)
        sns.lineplot(df1,x='logdose',y='edistance')
        plt.title(f'{cell_line},{drug}')
    return {d[0]:d[1] for d in zip(doses,dists)}

def compute_distance_sameplate_control(adatat1,distance_type='edistance',plotheatmap=False,plotdistances=False):
    distance = pt.tl.Distance(distance_type, obsm_key="X_pca")
    plates = np.unique(adatat1.obs['plate'])
    dss_dict = {}
    for p in plates:
        print(p)
        dss = compute_distance(adatat1[adatat1.obs['plate']==p,:], 
                               distance_type=distance_type,
                               plotheatmap=plotheatmap,
                              plotdistances=plotdistances)
        for k in dss:
            if k in dss_dict:
                dss_dict[k] = np.min([dss[k],dss_dict[k]])
            else:
                dss_dict[k] = dss[k]    
    return dss_dict


#################################################################
#################################################################

try:
    args = sys.argv[1:]
    cell_line = args[0]
    drug = args[1]
    work_folder = args[2]
    cl = cell_line
    dr = drug

    icafile = 'gs://tahoe100m_bycelllines/cellline_drug_analysis/'+cl+'_'+dr+'_ICA.h5ad'
    icafile_local = work_folder+cl+'_'+dr+'_ICA.h5ad'
    distfile = 'gs://tahoe100m_bycelllines/cellline_drug_analysis/'+cl+'_'+dr+'_distances.txt'
    distfile_local = work_folder+cl+'_'+dr+'_distances.txt'
    sigfile = 'gs://tahoe100m_bycelllines/cellline_drug_analysis/'+cl+'_'+dr+'_signature.pkl'
    sigfile_local = work_folder+cl+'_'+dr+'_signature.pkl'

      # first, we block the file from use by a parallel process
    print('Blocking file remotely:',icafile)
    with open(work_folder+'temp','w') as f:
        f.write('block')
    print('\t\t','copying to gs :',icafile)
    ec = os.system('gsutil -m cp '+work_folder+'temp "'+icafile+'"')
    print('\t\t','Exit code : ',ec)   
    
    # Loading files
    print('Loading files for...',cell_line,drug,work_folder)
    adata_cellline_collected = copy_celldrug_dataset(cell_line,drug,work_folder)
    genes_to_exclude = [g for g in adata_cellline_collected.var_names if (g.startswith('MT-')|(g=='MALAT1'))]
    print(f'{genes_to_exclude=}')
    adata_cellline = adata_cellline_collected[:,~adata_cellline_collected.var_names.isin(genes_to_exclude)].copy()
    del adata_cellline_collected
    adata_cellline.X = adata_cellline.X.todense()
    print('\t\t',adata_cellline)
    adata_cellline_pointer = adata_cellline
    adata_cellline = balance_control_measurement(adata_cellline)
    del adata_cellline_pointer
    print('\t\t',adata_cellline)

    # Preprocessing the file
    print('\t\t','Preprocessing',cl,dr)
    adatat1 = cellline_drug_file_preprocessing(adata_cellline)
    cc_regev = {'G1S_regev':cc.data._cc_markers.g1s_markers,
                'G2M_regev':cc.data._cc_markers.g2m_markers}
    calc_scores(adatat1,cc_regev)
    print('\t\t','Finished.')                    

    # Computing ICA
    print('\t\t','Computing ICA for',cl,dr)
    _compute_ica(adatat1,thr=0.4)
    Smatrix = adatat1.varm['P_dimRed']
    Amatrix = adatat1.obsm['X_dimRed']
    Metasamples_df = pd.DataFrame(data=Amatrix)
    Metasamples_df.index = adatat1.obs_names
    Metasamples_df.columns = [f'IC{i+1}' for i in range(Smatrix.shape[1])]
    Metasamples_adata = an.AnnData(Metasamples_df)
    Metasamples_adata.uns['S-phase_genes'] = list(adatat1.uns['S-phase_genes'])
    Metasamples_adata.uns['G2-M_genes'] = list(adatat1.uns['G2-M_genes'])
    Metasamples_adata.obs['G1S_regev'] = list(adatat1.obs['G1S_regev'])
    Metasamples_adata.obs['G2M_regev'] = list(adatat1.obs['G2M_regev'])
    Metasamples_adata.obs['plate'] = list(adatat1.obs['plate'])
    Metasamples_adata.obs['drugname_drugconc'] = list(adatat1.obs['drugname_drugconc'])
    Metasamples_adata.uns['scycle'] = list(adatat1.uns['scycle'])
    Metasamples_adata.uns['gene_means'] = np.mean(adatat1.X,axis=0)
    Metasamples_adata.uns['gene_names'] = list(adatat1.var_names)
    Metasamples_adata.uns['Metagenes'] = Smatrix 
    Metasamples_adata.write_h5ad(icafile_local,compression='gzip')

    # Computing distances
    print('\t\t','Computing distances for',cl,dr)
    de = compute_distance_sameplate_control(adatat1,distance_type='edistance')
    dw = compute_distance_sameplate_control(adatat1,distance_type='wasserstein')
    dm = compute_distance_sameplate_control(adatat1,distance_type='mmd')
    distances = {'edistance':de,'wassertein':dw,'mmd':dm}
    with open(distfile_local,'w') as f:
        f.write(str(distances))

    # Computing transcriptional signatures
    print('\t\t','Computing transcriptional signatures for',cl,dr)
    reference = "[('DMSO_TF', 0.0, 'uM')]"
    plates = np.unique(adatat1.obs['plate'])
    rgg_dict = {}
    for p in plates:
        print(p)
        adatas = adatat1[adatat1.obs['plate']==p,:]
        sc.tl.rank_genes_groups(adatas, groupby="drugname_drugconc", 
                            reference=reference, method="wilcoxon")
        rgg_dict[p] = adatas.uns['rank_genes_groups']
        del adatas
    with open(sigfile_local, 'wb') as handle:
        pickle.dump(rgg_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Copying the computed files
    print('\t\t','copying to gs :',icafile)
    os.system('gsutil -m cp "'+icafile_local+'" "'+icafile+'"')
    print('\t\t','copying to gs :',distfile)
    os.system('gsutil -m cp "'+distfile_local+'" "'+distfile+'"')
    print('\t\t','copying to gs :',sigfile)
    os.system('gsutil -m cp "'+sigfile_local+'" "'+sigfile+'"')


    # Cleaning
    os.system(f'rm "{icafile_local}"')
    os.system(f'rm "{distfile_local}"')
    os.system(f'rm "{sigfile_local}"')
    print('Executing',f'rm {work_folder}*"{cell_line}_{drug}"*.h5ad')
    os.system(f'rm {work_folder}*"{cell_line}_{drug}"*.h5ad')
    print('Executing',f'rm {work_folder}*"{cell_line}_{drug}"*.h5ad')
    os.system(f'rm {work_folder}*{cell_line}_DMSO*.h5ad')
    del adata_cellline
    del adatat1

    del Metasamples_df, Metasamples_adata
    
    gc.collect()

except Exception as e:
    print('\t\t','ERROR!!!! An exception occured for',cell_line,drug)
    print(traceback.format_exc())

