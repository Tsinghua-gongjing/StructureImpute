import pandas as pd
import numpy as np
import util

import argparse

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="ticks")
sns.set_context("poster")
plt.rcParams["font.family"] = "Helvetica"

from nested_dict import nested_dict
from scipy import stats

def compare(fragment, savefn):
    df = pd.read_csv(fragment, header=None, sep='\t')

    m1_ls,m2_ls,r_ls,p_ls=[],[],[],[]
    for i,j in zip(df[7],df[8]):
        shape1 = list(map(float,i.split(',')))
        shape2 = list(map(float,j.split(',')))
        r,p = stats.pearsonr(shape1, shape2)
        r_ls.append(r)
        p_ls.append(p)
        m1_ls.append(np.mean(shape1))
        m2_ls.append(np.mean(shape2))
    df['r'] = r_ls
    df['p'] = p_ls
    df['m1'] = m1_ls
    df['m2'] = m2_ls

    df['-log10(p)'] = -np.log10(df['p'])

    g = sns.jointplot(x='m1',y='m2',data=df,kind='kde', xlim=(0.0,0.5), ylim=(0.0,0.5), height=8, ratio=5)
    sns.regplot(df['m1'],df['m2'], scatter=False, ax=g.ax_joint)

    r,p = stats.pearsonr(df['m1'],df['m2'])
    s = 'R = {:.2f}\nP = {:.2e}\nN = {}'.format(r,p,df.shape[0])
    g.ax_joint.text(0.05, 0.9, s, ha='left', va='top', size=20, transform=g.ax_joint.transAxes)
    
    plt.tight_layout()
    plt.savefig(savefn)
    plt.close()
    
    fig,ax=plt.subplots(figsize=(10,6))
    ax.hist(df['r'], bins=100)
    ax.set_xlabel('Pearson correlation coefficient of fragment')
    ax.set_ylabel('# of fragment')
    plt.axvline(x=np.mean(df['r']), ymin=0, ymax=1, hold=None, ls='--', color='red')
    plt.tight_layout()
    plt.savefig(savefn.replace('.pdf','.hist.pdf'))
    plt.close()

def main():
    ####################################################################
    ### define parser of arguments
    parser = argparse.ArgumentParser(description='Compare two shape column in a fragment file')
    
    parser.add_argument('--fragment', type=str, help='fragment file')
    parser.add_argument('--savefn', type=str, help='Pdf file to save plot')
    
    # get args
    args = parser.parse_args()
    util.print_args(parser.description, args)
    
    compare(fragment=args.fragment, savefn=args.savefn)
    

if __name__ == '__main__':
    main()