from __future__ import print_function
import copy
import sys
import time
import argparse
import numpy as np
from nested_dict import nested_dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter

import structureimpute.models as arch
from structureimpute.engine import Engine
from structureimpute.utils import util
from structureimpute.dataset import StructureDataset

    
def main():
    # define parser of arguments
    parser = argparse.ArgumentParser(description='PyTorch for RNA SHAPE NULL value imputation')
    
    parser.add_argument('--model_name', type=str, default='None', \
                        help='Model name for the config')
    
    # train or predict direct
    parser.add_argument('--predict', action='store_true', 
                        help='Load saved model and predict only')
    parser.add_argument('--finetune', action='store_true', 
                        help='Load saved model and continue trainning with lr=lr/10')
    parser.add_argument('--load_model', type=str, help='Loaded model file')
    
    # training: hypyter parameter
    parser.add_argument('--arch', default="AllFusionNetMultiply", help='Model used')
    parser.add_argument('--optim', default="adam", help='optimizer used')
    parser.add_argument('--batch_size', type=int, default=400, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 10000)')
    parser.add_argument('--epochs', type=int, default=4000, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--weight_decay', type=float, default=0.0, metavar='WD',
                        help='Weight decay (default: 0.0)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--clip_norm', type=float, default=1.0, metavar='WD',
                        help='clip norm (default: 1.0)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--logdir', type=str, default='tfb',
                        help='tensorboard log directory')
    
    # train model: network paramter
    parser.add_argument('--train_type', type=str, default='trainHasNull_lossAll', 
                        help='Type of training data and loss function: \
                        [trainNoNull_lossAll, trainHasNull_lossAll, \
                        trainHasNull_lossNullOnly, trainHasNull_lossnull]')
    parser.add_argument('--monitor_val_loss', type=str, default='train_nonull_validate_nonull', 
                        help='Loss for monitor(selelct) model')
    parser.add_argument('--train_loss_null_weight', type=float, default=2.0, metavar='M',
                        help='Training loss function NULL part weight')
    parser.add_argument('--train_loss_func', type=str, default='mse', \
                        help='Type of loss fucntion for training')
    parser.add_argument('--early_stopping_num', type=int, default=40, metavar='N',
                        help='Early stop num') 
    
    parser.add_argument('--dropout', type=float, default=0.0, metavar='DR',
                        help='Dropout rate (default: 0.0)')
    
    # train model: LSTM
    parser.add_argument('--sequence_length', type=int, default=100, metavar='N',
                        help='Length of input RNA fragment')
    parser.add_argument('--sliding_length', type=int, default=100, metavar='N',
                        help='Sliding length while generating RNA fragments')
    parser.add_argument('--input_size', type=int, default=4, metavar='N',
                        help='Size of encoding RNA base')
    parser.add_argument('--lstm_hidden_size', type=int, default=128, metavar='N',
                        help='Hidden size in LSTM network')
    parser.add_argument('--lstm_num_layers', type=int, default=2, metavar='N',
                        help='Number of layers in LSTM network')
    parser.add_argument('--lstm_bidirectional', action='store_true', default=False,
                        help='Whether use bidirectional in LSTM or not?')
    parser.add_argument('--lstm_all_time', action='store_true', default=False,
                        help='Whether use output of all time from LSTM?')
    parser.add_argument('--lstm_output_time', type=int, default=-1,
                        help='LSTM output time step')
    parser.add_argument('--use_decode_loss', action='store_true', default=False,
                        help='Whether consider seq decode loss?')
    parser.add_argument('--attention_size', type=int, default=30, metavar='N',
                        help='Size used in attention')
    
    # train model: resiblock
    
    parser.add_argument('--channels1', type=int, default=8, metavar='N',
                        help='Channels of 1st conv')
    parser.add_argument('--channels2', type=int, default=32, metavar='N',
                        help='Channels of 2nd residual block')
    parser.add_argument('--channels3', type=int, default=64, metavar='N',
                        help='Channels of 3rd residual block')
    parser.add_argument('--channels4', type=int, default=128, metavar='N',
                        help='Channels of 4th residual block')
    parser.add_argument('--channels5', type=int, default=256, metavar='N',
                        help='Channels of 5th residual block')
    parser.add_argument('--channels6', type=int, default=128, metavar='N',
                        help='Channels of 6th residual block')
    parser.add_argument('--channels7', type=int, default=64, metavar='N',
                        help='Channels of 7th residual block')
    parser.add_argument('--channels8', type=int, default=32, metavar='N',
                        help='Channels of 8th residual block')
    parser.add_argument('--channels9', type=int, default=8, metavar='N',
                        help='Channels of 9th residual block')
    parser.add_argument('--channels10', type=int, default=8, metavar='N',
                        help='Channels of 10th residual block')
    parser.add_argument('--channels11', type=int, default=8, metavar='N',
                        help='Channels of 11th residual block')
    parser.add_argument('--channels12', type=int, default=8, metavar='N',
                        help='Channels of 12th residual block')
    
    parser.add_argument('--print_train_time', action='store_true', default=False,   
                        help='Whether print train time in each batch?')
    
    # file
    parser.add_argument('--filename_train', type=str, \
                        help='Path to training file')
    parser.add_argument('--filename_validation', type=str, \
                        help='Path to validation file')
    parser.add_argument('--output_valid_use_original', action='store_true', default=True, \
                        help='For valid position, keep orignal signal by default, else use predict signal')
    parser.add_argument('--out_featuremap', action='store_true', default=False, \
                        help='Whether output the inner feature maps')
    parser.add_argument('--filename_prediction', type=str, \
                        help='Path for saving predictions of validataion data set')
    parser.add_argument('--train_max_null_pct', type=float, default=1.0,
                        help='Keep fragments with pct(NULL)<=max_null_pct(default: 1.0) in train set')
    parser.add_argument('--validation_max_null_pct', type=float, default=1.0,
                        help='Keep fragments with pct(NULL)<=max_null_pct(default: 1.0) in validation set')
    
    # get args
    args = parser.parse_args()
    print(args)
    # fix seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # setting CUDA
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    ngpu = torch.cuda.device_count()
    print ('Available devices ', ngpu)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    if ngpu>0:
        args.batch_size      = args.batch_size*ngpu
        args.test_batch_size  = args.test_batch_size*ngpu
    # set dataloader
    if not args.predict:
        train_dataset = StructureDataset(fragment=args.filename_train)
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            shuffle=True, 
            **kwargs
        )
    
    validate_dataset = StructureDataset(fragment=args.filename_validation)
    validate_loader = torch.utils.data.DataLoader(
        dataset=validate_dataset,
        batch_size=args.test_batch_size,
        shuffle=False, 
        **kwargs
        )
    
    # define model & optimizer
    model = arch.model_entry(args.arch, args)

    engine = Engine(model, device, args)
    # prediction
    if args.predict:
        # load saved model
        print("Loading model:", args.load_model)
        pt = args.load_model
        checkpoint = torch.load(pt, map_location=torch.device('cpu'))
        util.load_module_state_dict(model, checkpoint, add=False, strict=True) # for DataParallel model
        
        # recover mode
        # util.load_state(args.load_model, model, optimizer=None)
        
        # dataparallel
        if ngpu >1 :
            model = nn.DataParallel(model, device_ids=range(ngpu))

        if use_cuda:
            model = model.cuda()
        if args.out_featuremap:
            validate_loss,prediction_all,featuremap = engine.validate(validate_loader)
            util.write_featuremap(featuremap, args.filename_prediction.replace('.txt', 'fmap.txt'), args.lstm_output_time)
        else:
            validate_loss,prediction_all = engine.validate(validate_loader)
        print("Load model and predict: loss:{}".format(validate_loss['train_nonull_validate_nonull']))
        if (args.output_valid_use_original):
            prediction_all = util.replace_predict_at_valid_with_original(prediction_all, validate_dataset.shape_matrix, validate_dataset.shape_true_matrix)
        with open(args.filename_prediction, 'w') as SAVEFN:
            for i in prediction_all:
                SAVEFN.write(','.join(map(str,i))+'\n')
        exit()
    
    # set optimizer 
    if args.optim == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    elif args.optim == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(optimizer)
    # fine-tune
    if args.finetune:
        print("Loading model:", args.load_model)
        pt = args.load_model
        checkpoint = torch.load(pt, map_location=torch.device('cpu'))
        util.load_module_state_dict(model, checkpoint, add=False, strict=True)
    
    # dataparallel
    if ngpu >1 :
        model = nn.DataParallel(model, device_ids=range(ngpu))
    if use_cuda:
        model = model.cuda()
    print(model)
    
    # initial loss
    validate_loss, prediction_all = engine.validate(validate_loader)
    print("Initial loss(only NULL): {}".format(
        validate_loss['train_hasnull_validate_onlynull']
        )
    )
    

    # training 
    writer = SummaryWriter(log_dir=args.logdir)
    train_start_time = time.time()
    min_loss,min_epoch,min_prediction = 100,0,0
    for epoch in range(1, args.epochs + 1):
        
        if args.train_type == 'trainHasNull_lossAll':
            train_loss,train_loss_nonnull,train_loss_null = engine.train(train_loader, optimizer, epoch)
        else:
            train_loss = engine.train(train_loader, optimizer, epoch)
        if args.out_featuremap:
            validate_loss,prediction_all,featuremap = engine.validate(validate_loader)
        else:
            validate_loss,prediction_all = engine.validate(validate_loader)
        
        monitor_val_loss = validate_loss[args.monitor_val_loss]
        
        is_best = monitor_val_loss < min_loss
        best_val_loss = min(monitor_val_loss, min_loss)
        
        if is_best:
            min_loss = validate_loss[args.monitor_val_loss]
            min_epoch = epoch
            min_prediction = prediction_all
            
            with open(args.filename_prediction, 'w') as SAVEFN:
                print("Write prediction: epoch:{}, loss:{}".format(min_epoch, min_loss))
                for i in min_prediction:
                    SAVEFN.write(','.join(map(str,i))+'\n')
            
            best_model = copy.deepcopy(model)
        else:
            if epoch - min_epoch > args.early_stopping_num:
                print('Early stopping!')
                break
        
        writer.add_scalar('Train/loss', train_loss, epoch)
        if args.train_type == 'trainHasNull_lossAll':
            writer.add_scalar('Train/loss_nonnull', train_loss_nonnull, epoch)
            writer.add_scalar('Train/loss_null', train_loss_null, epoch)
        writer.add_scalar('Validation/train_nonull_validate_nonull', 
                          validate_loss['train_nonull_validate_nonull']*args.batch_size, epoch)
        writer.add_scalar('Validation/train_hasnull_validate_nonull', 
                          validate_loss['train_hasnull_validate_nonull']*args.batch_size, epoch)
        writer.add_scalar('Validation/train_hasnull_validate_hasnull', 
                          validate_loss['train_hasnull_validate_hasnull']*args.batch_size, epoch)
        writer.add_scalar('Validation/train_hasnull_validate_onlynull', 
                          validate_loss['train_hasnull_validate_onlynull']*args.batch_size, epoch)
        
    writer.close()
    train_end_time = time.time()
    util.timer(start=train_start_time, end=train_end_time, description='Training time')
    print("Write prediction: epoch:{}, loss:{}".format(min_epoch, min_loss))
    
    # saving model
    torch.save(best_model.state_dict(), 'model.pt'))
        
if __name__ == '__main__':
    main()
