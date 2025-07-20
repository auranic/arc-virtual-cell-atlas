import sys
import os
import pickle
import traceback
from guppy import hpy
import subprocess
import random
h = hpy()

def check_if_gsfile_exists(url):
    result = subprocess.run(['gsutil','ls',url], capture_output=True, text=True)
    out = result.stdout[:-1]
    return out==url

#################################################################
#################################################################

result = subprocess.run(['gsutil','ls','gs://tahoe100m_bycelllines/panel_cell_drug/'], capture_output=True, text=True)
out = result.stdout
lst = out.split('\n')
cell_lines = list(set([l.split('/')[-1].split('_')[1]+'_'+l.split('/')[-1].split('_')[2] for l in lst if l.endswith('h5ad')]))
print('Cell lines = ',len(cell_lines))
drug_names = list(set([l.split('/')[-1].split('_')[3] for l in lst if l.endswith('h5ad')]))
print('Number of drugs =',len(drug_names))

cell_lines.sort()
random.shuffle(cell_lines)
drug_names.sort()

existing_files = subprocess.run(['gsutil','ls','gs://tahoe100m_bycelllines/cellline_drug_analysis/'], capture_output=True, text=True).stdout.split('\n')
# quick check for existing files
file_preexist = []
for cl in cell_lines:
    for dr in drug_names:
        icafile = 'gs://tahoe100m_bycelllines/cellline_drug_analysis/'+cl+'_'+dr+'_ICA.h5ad'
        if icafile in existing_files:
            file_preexist.append(True)
        else:
            file_preexist.append(False)

work_folder = './data/cellline_drug/'

k = 0
for ic,cell_line in enumerate(cell_lines):
    print(ic+1,len(cell_lines),cell_line)
    for id,drug in enumerate(drug_names):
        print('======================================\n\n\t','cl:',ic,len(cell_lines),'dg:',id,len(drug_names),drug)
        icafile = 'gs://tahoe100m_bycelllines/cellline_drug_analysis/'+cell_line+'_'+drug+'_ICA.h5ad'
        if not file_preexist[k]:
            if not check_if_gsfile_exists(icafile):
                #print(h.heap())
                #subprocess.run()           
                os.system('python -W ignore compute_cellline_drug_worker.py '+cell_line+' "'+drug+'" '+work_folder)
            else:
                print('File seems to be already processed')
        else:
            print('File seems to be already processed')
        k+=1

