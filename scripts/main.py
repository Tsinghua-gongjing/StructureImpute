from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from nested_dict import nested_dict
from tensorboardX import SummaryWriter
import copy
import sys
import time

import util
import models as arch
import train
import validate
from torchviz import make_dot

import matplotlib as mpl
mpl.use('Agg')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ShapeDataset():
    def __init__(self, fragment, fragment_len, dataset, input_size, max_null_pct):
        self.fragment = fragment
        self.fragment_len = fragment_len
        self.dataset = dataset
        self.feature_size = input_size
        self.max_null_pct = max_null_pct
        self.seq_matrix,self.shape_matrix,self.shape_true_matrix = util.fragment_to_format_data(
            fragment=self.fragment, fragment_len=self.fragment_len, dataset=self.dataset, feature_size=self.feature_size, max_null_pct=self.max_null_pct)
        self.seq_matrix = torch.from_numpy(self.seq_matrix[0:])
        self.shape_matrix = torch.from_numpy(self.shape_matrix[0:])
        self.shape_true_matrix = torch.from_numpy(self.shape_true_matrix[0:])
        self.shape_matrix_null_mask = torch.ones_like(self.shape_matrix)
        self.shape_matrix_null_mask[self.shape_matrix<0] = 0 
        self.shape_true_matrix_null_mask = torch.ones_like(self.shape_true_matrix)
        self.shape_true_matrix_null_mask[self.shape_true_matrix<0] = 0
        self.len = self.seq_matrix.shape[0]
        # print("[Check data len] {}: {}".format(self.fragment, self.len))
        
    def __getitem__(self, index):
        return self.seq_matrix[index], \
               self.shape_matrix[index], \
               self.shape_matrix_null_mask[index], \
               self.shape_true_matrix[index], \
               self.shape_true_matrix_null_mask[index]
    
    def __len__(self):
        return self.len
    
def main():
    ####################################################################
    ### define parser of arguments
    parser = argparse.ArgumentParser(description='PyTorch for RNA SHAPE NULL value imputation')
    
    parser.add_argument('--model_name', type=str, default='None', \
                        help='Model name for the config')
    
    # train or predict direct
    # parser.add_argument('--load_model_and_predict', type=util.str2bool, nargs='?', const=True,
    #                     default=False, help='Load saved model and predict only')
    parser.add_argument('--load_model_and_predict', action='store_true', 
                        help='Load saved model and predict only')
    parser.add_argument('--load_model_and_continue_train', action='store_true', 
                        help='Load saved model and continue trainning with lr=lr/10')
    parser.add_argument('--load_model_and_continue_train_with_sgd', action='store_true', 
                        help='Load saved model and continue trainning with SGD optimizer')
    parser.add_argument('--loaded_pt_file', type=str, help='Loaded .pt file to predict')
    
    # training: hypyter parameter
    parser.add_argument('--arch', default="AllFusionNetMultiply", help='Model used')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test_batch_size', type=int, default=10000, metavar='N',
                        help='input batch size for testing (default: 10000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--weight_decay', type=float, default=0.0, metavar='WD',
                        help='Weight decay (default: 0.0)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    
    # train model: network paramter
    # train model: loss function
    parser.add_argument('--train_type', type=str, default='trainHasNull_lossAll', 
                        help='Type of training data and loss function: \
                        [trainNoNull_lossAll, trainHasNull_lossAll, \
                        trainHasNull_lossNullOnly, trainHasNull_lossnull]')
    parser.add_argument('--monitor_val_loss', type=str, default='train_nonull_validate_nonull', 
                        help='Loss for monitor(selelct) model')
    parser.add_argument('--train_loss_null_weight', type=float, default=1.0, metavar='M',
                        help='Training loss function NULL part weight')
    parser.add_argument('--train_loss_func', type=str, default='mse', \
                        help='Type of loss fucntion for training')
    parser.add_argument('--early_stopping_num', type=int, default=100, metavar='N',
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
    parser.add_argument('--use_residual', action='store_true', default=False,   
                        help='Whether use residual blocks before LSTM or not?')
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
    
    # save model
    parser.add_argument('--save_model', action='store_true', default=False,
                        help='For saving the trained model')
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
    parser.add_argument('--logfile', type=str, \
                        help='Path to log file')
    parser.add_argument('--train_max_null_pct', type=float, default=1.0,
                        help='Keep fragments with pct(NULL)<=max_null_pct(default: 1.0) in train set')
    parser.add_argument('--validation_max_null_pct', type=float, default=1.0,
                        help='Keep fragments with pct(NULL)<=max_null_pct(default: 1.0) in validation set')
    
    # get args
    args = parser.parse_args()
    ####################################################################
    
    log = open(args.logfile, 'w')
    sys.stdout = log
    
    ####################################################################
    ### setting CUDA
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    ngpu = torch.cuda.device_count()
    print ('Available devices ', ngpu)
    # torch.manual_seed(args.seed)
    
    # random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    # print(device)
    # exit() # check whether GPU is used
    ####################################################################

    
    ####################################################################
    ### load train/validate data set
    train_dataset = ShapeDataset(fragment=args.filename_train, 
                                 fragment_len=args.sequence_length, 
                                 dataset='validation', input_size=args.input_size,
                                max_null_pct=args.train_max_null_pct)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=True, **kwargs)
    
    validate_dataset = ShapeDataset(fragment=args.filename_validation, 
                                    fragment_len=args.sequence_length, 
                                    dataset='validation', input_size=args.input_size,
                                   max_null_pct=args.validation_max_null_pct)
    validate_loader = torch.utils.data.DataLoader(dataset=validate_dataset,
                                              batch_size=args.test_batch_size,
                                              shuffle=False, **kwargs)
    ####################################################################
    
    
    ####################################################################
    ### define model & optimizer
    # model = getattr(arch, args.arch)(args)
    model = arch.model_entry(args.arch, args)
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    ####################################################################
    ### load model + predict + save predictions
    if args.load_model_and_predict:
        # load saved model
        # pt = args.loaded_pt_file
        # checkpoint = torch.load(pt)
        # model.load_state_dict(checkpoint, strict=False)
        util.load_module_state_dict(model, torch.load(args.loaded_pt_file), add=False, strict=True) # only load model
        
        # recover mode
#         util.load_state(args.loaded_pt_file, model, optimizer=None)
        
        # dataparallel
        if ngpu >1 :
            model = nn.DataParallel(model, device_ids=range(ngpu))
        model = model.cuda()
        for name, param in model.named_parameters():
                if param.requires_grad:
                    print(name, param.data)
        if args.out_featuremap:
            validate_loss,prediction_all,featuremap = validate.validate_LSTM(args, model, device, validate_loader)
            util.write_featuremap(featuremap, args.filename_prediction.replace('.txt', 'fmap.txt'), args.lstm_output_time)
        else:
            validate_loss,prediction_all = validate.validate_LSTM(args, model, device, validate_loader)
        print("Load model and predict: loss:{}".format(validate_loss['train_nonull_validate_nonull']))
        if (args.output_valid_use_original):
            prediction_all = util.replace_predict_at_valid_with_original(prediction_all, validate_dataset.shape_matrix, validate_dataset.shape_true_matrix)
        with open(args.filename_prediction, 'w') as SAVEFN:
            for i in prediction_all:
                SAVEFN.write(','.join(map(str,i))+'\n')
        exit()
    ####################################################################    
    

    ####################################################################
    ### load model + continue train
    if args.load_model_and_continue_train:
        # load saved model
        # pt = args.loaded_pt_file
        # checkpoint = torch.load(pt)
        # model.load_state_dict(checkpoint, strict=False)
        # load_module_state_dict(model, checkpoint, add=False, strict=True)
        # args.lr = args.lr/10
        
        if args.load_model_and_continue_train_with_sgd:
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            
        best_val_loss, load_epoch = util.load_state(args.loaded_pt_file, model, optimizer=optimizer)
        
    ####################################################################
    
    print(optimizer)
    
    # dataparallel
    if ngpu >1 :
        model = nn.DataParallel(model, device_ids=range(ngpu))
    model = model.cuda()
    print(model)
    
    
    ####################################################################
    
    
    ####################################################################
    ### training and monitor loss with tensorboardx
    x = torch.rand(10, 1, 100, args.input_size).cuda()
    y = torch.rand(10, 1, 100, 1).cuda()
    Y = model(x,y)
    # g = make_dot(Y, params=dict(list(model.named_parameters()) + [('x', x)]))
    # g.view()
    
    # w2 = SummaryWriter()
    # w2.add_graph(model, (x,y), verbose=False)
    
    writer = SummaryWriter()
    # writer.add_graph(model, (x,y), verbose=False)
    
    train_start_time = time.time()
    min_loss,min_epoch,min_prediction = 100,0,0
    for epoch in range(1, args.epochs + 1):
        if args.load_model_and_continue_train: epoch += load_epoch
        # util.adjust_learning_rate(optimizer, epoch, args)
        if args.train_type == 'trainHasNull_lossAll':
            train_loss,train_loss_nonnull,train_loss_null = train.train_LSTM(args, model, device, train_loader, optimizer, epoch)
        else:
            train_loss = train.train_LSTM(args, model, device, train_loader, optimizer, epoch)
        if args.out_featuremap:
            validate_loss,prediction_all,featuremap = validate.validate_LSTM(args, model, device, validate_loader)
        else:
            validate_loss,prediction_all = validate.validate_LSTM(args, model, device, validate_loader)
        
        monitor_val_loss = validate_loss[args.monitor_val_loss]
        
        is_best = monitor_val_loss < min_loss
        best_val_loss = min(monitor_val_loss, min_loss)
        util.save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'train_loss': train_loss,
            'best_val_loss': best_val_loss,
        }, is_best, args.filename_prediction.replace('.txt', '.pt'))
        
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
        
        # if (args.save_model):
            # torch.save(model.state_dict(), args.filename_prediction.replace('.txt', '.latest.pt'))
        
    writer.close()
    train_end_time = time.time()
    util.timer(start=train_start_time, end=train_end_time, description='Training time')
    
    print("Write prediction: epoch:{}, loss:{}".format(min_epoch, min_loss))
    ####################################################################
    
    
    ####################################################################
    # saving model
    if (args.save_model):
        torch.save(best_model.state_dict(), args.filename_prediction.replace('.txt', '.pt'))
    ####################################################################
    
    log.close()
        
if __name__ == '__main__':
    main()
