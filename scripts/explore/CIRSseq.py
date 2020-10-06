from pyfasta import Fasta


def loadWigHash(filename):
    curchrm = ""
    returnval = {}
    fo = open(filename, "r")
    for line in fo:
        line = line.strip()
        if line.startswith("variableStep"):
                terms = line.split(" ")
                if len(terms) == 2:
                    # print "chr: %s"%(terms[1])
                    curchrm = terms[1].split('=')[1].split('|')[0]
                    if curchrm not in returnval:
                        returnval[curchrm] = {}
        else:
            vals = line.rstrip().split()
            if len(vals) == 2:
                returnval[curchrm][int(vals[0])] = float(vals[1])
    fo.close()
    print("read num of chrs: %s"%(len(returnval)))
    print(returnval['ENSMUST00000174924'])
    return returnval

def retrieve_one_tx():
    returnval = loadWigHash('/home/gongjing/project/shape_imputation/data/CIRSseq/GSE54106_CIRS-seq_Reactivity_combined.wig')
    tx_id = 'ENSMUST00000175096'
    tx_len = 319
    savefn = '/home/gongjing/project/shape_imputation/data/CIRSseq/CIRSseq.%s.out'%(tx_id)

    if tx_id in returnval:
        with open(savefn, 'w') as SAVEFN:
            reactivity_ls = []
            for i in range(1, tx_len+1):
                if i in returnval[tx_id]:
                    reactivity_ls.append(returnval[tx_id][i])
                else:
                    reactivity_ls.append('NULL')
            SAVEFN.write('\t'.join(map(str, [tx_id, tx_len, '0']+reactivity_ls))+'\n')

def wig_to_out():
    wig = '/home/gongjing/project/shape_imputation/data/CIRSseq/GSE54106_CIRS-seq_Reactivity_combined.wig'
    returnval = loadWigHash(wig)
    fa = '/home/gongjing/project/shape_imputation/data/CIRSseq/cirs.fa'
    fa_dict1 = Fasta(fa)
    fa_dict = {}
    for i,j in fa_dict1.items():
        fa_dict[i.split('|')[0]] = j[0:]

    savefn = wig.replace('.wig', '.out')
    with open(savefn, 'w') as SAVEFN:
        for tx_id in returnval:
            if tx_id in fa_dict:
                tx_len = len(fa_dict[tx_id][0:])
                
                reactivity_ls = []
                for i in range(1, tx_len+1):
                    if i in returnval[tx_id]:
                        reactivity_ls.append(returnval[tx_id][i])
                    else:
                        reactivity_ls.append('NULL')
                SAVEFN.write('\t'.join(map(str, [tx_id, tx_len, '0']+reactivity_ls))+'\n')
    

if __name__ == '__main__':
#     main()
    wig_to_out()