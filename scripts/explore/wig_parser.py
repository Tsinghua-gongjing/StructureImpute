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
    print "read num of chrs: %s"%(len(returnval))
    print returnval['ENSMUST00000174924']
    return returnval

def main():
    returnval = loadWigHash('/Share/home/zhangqf5/gongjing/Kethoxal_RNA_structure/data/CIRSseq/GSE54106_CIRS-seq_Reactivity_combined.wig')
    tx_id = 'ENSMUST00000175096'
    tx_len = 319
    savefn = '/Share/home/zhangqf5/gongjing/Kethoxal_RNA_structure/data/CIRSseq/CIRSseq.%s.out'%(tx_id)

    if returnval.has_key(tx_id):
        with open(savefn, 'w') as SAVEFN:
            reactivity_ls = []
            for i in xrange(1, tx_len+1):
                if returnval[tx_id].has_key(i):
                    reactivity_ls.append(returnval[tx_id][i])
                else:
                    reactivity_ls.append('NULL')
            print >>SAVEFN, '\t'.join(map(str, [tx_id, tx_len, '0']+reactivity_ls))


if __name__ == '__main__':
    main()