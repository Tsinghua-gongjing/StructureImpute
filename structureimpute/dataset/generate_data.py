
import numpy as np
import argparse

from structureimpute.utils import util



    
def main():
    ####################################################################
    ### define parser of arguments
    parser = argparse.ArgumentParser(description='PyTorch for RNA SHAPE NULL value imputation')
    
    # train model: LSTM
    parser.add_argument('--sequence_length', type=int, default=100, metavar='N',
                        help='Length of input RNA fragment')
    parser.add_argument('--feature_size', type=int, default=4, metavar='N',
                        help='Size of encoding RNA base')
    # file
    parser.add_argument('--filename_train', type=str, \
                        help='Path to training file')
    parser.add_argument('--filename_validation', type=str, \
                        help='Path to validation file')
    parser.add_argument('--train_max_null_pct', type=float, default=1.0,
                        help='Keep fragments with pct(NULL)<=max_null_pct(default: 1.0) in train set')
    parser.add_argument('--validation_max_null_pct', type=float, default=1.0,
                        help='Keep fragments with pct(NULL)<=max_null_pct(default: 1.0) in validation set')
    
    # get args
    args = parser.parse_args()

    seq, struct, struct_true = util.fragment_to_format_data(
        fragment=args.filename_train, 
        fragment_len=args.sequence_length, 
        dataset='validation', 
        feature_size=args.feature_size,
        max_null_pct=args.train_max_null_pct
    )

    np.savez_compressed(
        args.filename_train+".npz",
        seq=seq, 
        struct=struct, 
        struct_true=struct_true
    )


    #seq, struct, struct_true = util.fragment_to_format_data(
    #    fragment=args.filename_validation, 
    #    fragment_len=args.sequence_length, 
    #    dataset='validation', 
    #    feature_size=args.feature_size,
    #    max_null_pct=args.validation_max_null_pct)

    #np.savez_compressed(
    #        args.filename_validation+".npz",
    #        seq=seq, 
    #        struct=struct, 
    #        struct_true=struct_true
    #    )

if __name__ == "__main__":
    main()
