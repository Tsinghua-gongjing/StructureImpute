import os
import argparse

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="ticks")
sns.set_context("poster")
plt.rcParams["font.family"] = "Helvetica"

from nested_dict import nested_dict
import util
import pandas as pd

def dir_files(d):
    return os.listdir(d)

def read_rfam(f):
    d = {}
    with open(f, 'r') as F:
        for n,line in enumerate(F):
            if n >= 5: continue
            line = line.strip()
            if line.startswith('#'):
                arr = line.split(':')
                d[arr[0].replace('#','').replace(':','')] = arr[1].rstrip()
            if n == 3: d['seq'] = line
            if n == 4: d['dot'] = line
    return d

def read_dir_rfam(d, savefn):
    files = dir_files(d)
    
    dir_rfam_dict = nested_dict(2, int)
    for file in files:
        file_path = d + '/' + file
        file_dict = read_rfam(file_path)
        for i,j in file_dict.items():
            dir_rfam_dict[file][i] = j.replace(' ','')
        
    dir_rfam_df = pd.DataFrame.from_dict(dir_rfam_dict, orient='index')
    print(dir_rfam_df.head())

    cols = ['Name', 'Length', 'seq', 'dot']
    dir_rfam_df[cols].to_csv(savefn, header=True, index=False, sep='\t')
    
    with open(savefn.replace('.txt', '.fa'), 'w') as FA:
        for i,j in dir_rfam_dict.items():
            FA.write('>'+i+'\n')
            FA.write(j['seq']+'\n')

def main():
    ####################################################################
    ### define parser of arguments
    parser = argparse.ArgumentParser(description='Parse dataset of bpRNA in SPOT-RNA project')
    
    parser.add_argument('--d', type=str, help='Dir of bpRNA')
    parser.add_argument('--savefn', type=str, help='File to save parsed info')
    
    # get args
    args = parser.parse_args()
    util.print_args(parser.description, args)
    
    read_dir_rfam(d=args.d, savefn=args.savefn)
    

if __name__ == '__main__':
    main()