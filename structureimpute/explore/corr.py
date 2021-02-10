import sys
import pandas as pd
from scipy import stats

f = sys.argv[1]
df_tx = pd.read_csv(f,sep='\t')
import pdb;pdb.set_trace()
tx_r,tx_p = stats.pearsonr(df_tx['True'], df_tx['Predict'])
print("{:.3f}\t{:.3e}".format(tx_r,tx_p))
