import os
import gc
import subprocess
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats.mstats import winsorize
from scipy import stats
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from collections import Counter
import scanpy as sc
import anndata as an
import pandas as pd
import numpy as np
import sys
#sys.path.append('/mnt/c/MyPrograms/__github/scycle/')
sys.path.append('./code/scycle/')
import scycle as cc

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

def preprocessing_without_pooling(adata):
    if not Already_Log_Transformed:
        sc.pp.log1p(adata)
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
    sc.tl.pca(adata,n_comps=number_of_pcs)
    return adata

# pooling procedure
def pooling_procedure(adata):
    if n_neighbours_for_pooling>0:    
        adata_work = adata_orig.copy()
        preprocessing_without_pooling(adata)
        sc.tl.pca(adata,n_comps=number_of_pcs)
        X_pca = adata.obsm['X_pca']
        adata = smooth_adata_by_pooling(adata_work,X_pca,n_neighbours=n_neighbours_for_pooling)
    return adata

def preprocessing_dataset(adata):
    adata = preprocessing_without_pooling(adata)    
    sc.tl.pca(adata,n_comps=number_of_pcs)
    display(adata)
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
    cc.tl.dimensionality_reduction(adata,method='ica')

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

work_folder = './data/'

def check_if_gsfile_exists(url):
    result = subprocess.run(['gsutil','ls',url], capture_output=True, text=True)
    out = result.stdout[:-1]
    return out==url
    

import subprocess

task = 'pseudobulk'
task = 'decomposebydrug'
minnumberofcellsfordrugresponse = 500

print('Current folder:')
subprocess.run('pwd')

print('Getting list of files to proceed....')
result = subprocess.run(['gsutil','ls','gs://tahoe100m_bycelllines/'], capture_output=True, text=True)
out = result.stdout
lst = out.split('\n')
print('Got',len(lst),'files')


existing_files = []
file_preexist = []

if task=='pseudobulk':
    existing_files = subprocess.run(['gsutil','ls','gs://tahoe100m_analysis_pseudobulk/pseudobulk_files/'], capture_output=True, text=True).stdout.split('\n')
    # quick check for existing files
    file_preexist = []
    for f in lst:
        gsbulkfile = 'gs://tahoe100m_analysis_pseudobulk/pseudobulk_files/'+f.split('/')[-1][:-5]+'_bulk.tsv'
        if gsbulkfile in existing_files:
            file_preexist.append(True)
        else:
            file_preexist.append(False)
if task=='decomposebydrug':
    existing_files = subprocess.run(['gsutil','ls','gs://tahoe100m_bycelllines/panel_cell_drug/'], capture_output=True, text=True).stdout.split('\n')
    # quick check for existing files
    file_preexist = []
    for f in lst:
        drugcellnumberfile = 'gs://tahoe100m_bycelllines/panel_cell_drug/'+f.split('/')[-1][:-5]+'_drug_cellnumbers.tsv'
        if drugcellnumberfile in existing_files:
            file_preexist.append(True)
        else:
            file_preexist.append(False)


for i,l in enumerate(lst):
    if l.endswith('.h5ad'):
        filename = l        
        infile = filename
        file = f'{work_folder}{filename}'
        localfile = work_folder+infile.split('/')[-1]
        targetfile = ''
        if task=='pseudobulk':
            gsbulkfile = 'gs://tahoe100m_analysis_pseudobulk/pseudobulk_files/'+infile.split('/')[-1][:-5]+'_bulk.tsv'
            gsncellsfile = 'gs://tahoe100m_analysis_pseudobulk/cell_numbers/'+infile.split('/')[-1][:-5]+'_numberofcells.txt'
            targetfile = gsbulkfile
        if task=='decomposebydrug':
            drugcellnumberfile = 'gs://tahoe100m_bycelllines/panel_cell_drug/'+infile.split('/')[-1][:-5]+'_drug_cellnumbers.tsv'
            targetfile = drugcellnumberfile

        print(i+1,'Processing ',file)

        #if not os.path.exists(localfile[:-5]+'_bulk.tsv'):
        if not file_preexist[i]:
            if not check_if_gsfile_exists(targetfile):

                # first, we block the file from use by a parallel process
                with open(work_folder+'temp','w') as f:
                    f.write('block')
                print('copying to gs :',targetfile)
                os.system('gsutil -m cp '+work_folder+'temp '+targetfile)

                cmdline = f'gsutil -m cp {infile} {work_folder}'
                print('Executing: '+cmdline)
                ec = os.system(cmdline)
                print('Exit code : ',ec)
                if ec==0:
                    print('Loading ',localfile)
                    adata = sc.read_h5ad(localfile)
                    fn = localfile.split('/')[-1]
                    plate = fn.split('_')[0]
                    cell_line = fn.split('_')[1]+'_'+fn.split('_')[2][:-5]
                    print(f'{plate=},{cell_line=}')

                    if task=='pseudobulk':
                        # Extract number of cells in each condition and the pseudobulk
                        import ast
                        drugs = list(set(adata.obs['drugname_drugconc']))
                        print(drugs)
                        number_of_drugs = len(drugs)
                        print(f'{number_of_drugs=}')
                        bulk_vals = {}
                        with open(localfile[:-5]+'_numberofcells.txt','w') as f:
                            f.write('MEASUREMENT\tNUMBER_OF_CELLS\tNUMBER_OF_READS\n')
                            for d in drugs:
                                dt = ast.literal_eval(d)[0]
                                drugname = str(dt[0])+'__'+str(dt[1])
                                adf = adata[adata.obs['drugname_drugconc']==d,:]
                                print(drugname,len(adf))
                                f.write(drugname+'\t'+str(len(adf))+'\t'+str(int(adf.X.sum()))+'\n')
                                bulk_vals[drugname] = adf.X.sum(axis=0)[0,:].A1
                        print('copying to gs :',gsncellsfile)
                        os.system('gsutil -m cp '+localfile[:-5]+'_numberofcells.txt '+gsncellsfile)
                        df = pd.DataFrame(data=bulk_vals)
                        df.index = adata.var_names
                        df.to_csv(localfile[:-5]+'_bulk.tsv',sep='\t')
                        print('copying to gs :',targetfile)
                        os.system('gsutil -m cp '+localfile[:-5]+'_bulk.tsv '+targetfile)
                        del bulk_vals
                        del df
                        os.system(f'rm {localfile}')  
                        del adata   


                    if task=='decomposebydrug':
                        import ast
                        drugs = list(set(adata.obs['drug']))
                        cellnumbers = []
                        drugconclist = []
                        decomposed_filenames = []
                        k = 0
                        for drug_name in drugs:
                            if '/' in drug_name:
                                drug_name = drug_name.replace('/','-')
                            adata_drug = adata[adata.obs['drug']==drug_name,:]
                            conc_vals = np.unique(list(adata_drug.obs['drugname_drugconc']))
                            for cv in conc_vals:
                                cvt = ast.literal_eval(cv)
                                adata_drug_conc =  adata_drug[adata_drug.obs['drugname_drugconc']==cv,:]
                                if len(adata_drug_conc)>minnumberofcellsfordrugresponse:
                                    conc = str(cvt[0][1])
                                    print(k+1,drug_name,conc)
                                    dfn = plate+'_'+cell_line+'_'+drug_name+'_'+conc+'.h5ad'
                                    adata_drug_conc.write_h5ad(work_folder+dfn,compression='gzip')
                                    decomposed_filenames.append(dfn)
                                    cellnumbers.append(len(adata_drug_conc))
                                    if len(conc_vals)==1:
                                        drugconclist.append(drug_name)
                                    else:
                                        drugconclist.append(drug_name+'#'+conc)
                                    del adata_drug_conc
                                    k+=1
                        os.system(f'rm {localfile}')  
                        del adata   

                        #if len(decomposed_filenames)==0:
                        #    print('deleting from gs :',localfile[:-5]+'_numberofcells.txt '+drugcellnumberfile)    
                        #    os.system('gsutil rm gs://tahoe100m_bycelllines/panel_cell_drug/'+localfile[:-5]+'_numberofcells.txt '+drugcellnumberfile)


                        print('copying to gs :',localfile[:-5]+'*.h5ad')
                        os.system('gsutil -m cp '+localfile[:-5]+'*.h5ad '+'gs://tahoe100m_bycelllines/panel_cell_drug/')
                        os.system('rm '+localfile[:-5]+'*.h5ad')

                        df = pd.DataFrame(data={'DRUG':drugconclist,'CELL_NUMBERS':cellnumbers})
                        df.to_csv(localfile[:-5]+'_drug_cellnumbers.tsv',sep='\t',index=False)
                        print('copying to gs :',drugcellnumberfile)
                        os.system('gsutil -m cp '+localfile[:-5]+'_drug_cellnumbers.tsv '+drugcellnumberfile)

                        os.system('rm '+localfile[:-5]+'_drug_cellnumbers.tsv')


                    gc.collect()
                    
            else:
                print('File seems to be already processed')
        else:
            print('File seems to be already processed')
