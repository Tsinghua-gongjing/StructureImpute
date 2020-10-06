from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import copy
from nested_dict import nested_dict

import util

def validate_LSTM(args, model, device, validate_loader, save_prediction=None):
    model.eval()
    # validate_loss = 0
    # random_null_loss = 0
    loss = nested_dict(1, int)
    # criterion = nn.SmoothL1Loss()
    criterion = nn.MSELoss(reduction='mean')
    prediction = []
    featuremap = [[],[],[]]
    with torch.no_grad():
        for seq_matrix, shape_matrix, shape_null_mask,shape_true_matrix,shape_true_matrix_null_mask in validate_loader:
            if args.print_train_time:
                print("[Check shape] validate shape_matrix: {},shape_null_mask: {}, shape_true_matrix: {}, shape_true_matrix_null_mask: {}".format(shape_matrix.shape, shape_null_mask.shape, shape_true_matrix.shape, shape_true_matrix_null_mask.shape))
                print("[Check null sum] validate: shape_null_mask {}, shape_true_matrix_null_mask {}".format(shape_null_mask.sum(), shape_true_matrix_null_mask.sum()))
            
            seq_matrix = seq_matrix.float().reshape(-1, args.sequence_length, args.input_size).cuda()
            shape_matrix = shape_matrix.float().cuda()
            shape_true_matrix = shape_true_matrix.float().cuda()
            
            # 对全为valid shape的矩阵，+seq 送到模型进行预测
            if args.use_decode_loss:
                output,x_decode = model(seq_matrix, shape_true_matrix.reshape(-1, args.sequence_length, 1))
            else:
                if args.out_featuremap:
                    output,fmap = model(seq_matrix, shape_matrix.reshape(-1, args.sequence_length, 1))
                else:
                    output = model(seq_matrix, shape_true_matrix.reshape(-1, args.sequence_length, 1))
            if args.print_train_time: print("[Check shape] validate output: {}".format(output.shape))
            
            if args.lstm_all_time:
                # shape_null_mask = shape_null_mask.view(shape_null_mask.shape[0], -1, args.sequence_length)
                # shape_true_matrix = shape_true_matrix.view(shape_true_matrix.shape[0], -1, args.sequence_length)
                # shape_true_matrix_null_mask = shape_true_matrix_null_mask.view(shape_true_matrix_null_mask.shape[0], -1, args.sequence_length)
                # shape_null_mask = shape_null_mask.repeat(1, args.sequence_length, 1)
                # shape_true_matrix = shape_true_matrix.repeat(1, args.sequence_length, 1)
                # shape_true_matrix_null_mask = shape_true_matrix_null_mask.repeat(1, args.sequence_length, 1)
                
                output = output[:,-1,:]
                # print("[Check shape] validate output: {}".format(output.shape))
            
            loss_train_nonull_validate_nonull = criterion(output[shape_true_matrix_null_mask>0], 
                                                          shape_true_matrix[shape_true_matrix_null_mask>0]) # 只对非null值计算loss
            loss["train_nonull_validate_nonull"] += loss_train_nonull_validate_nonull
#             loss_calc = nn.MSELoss()
#             loss_calc_out = loss_calc(output[shape_true_matrix_null_mask>0],shape_true_matrix[shape_true_matrix_null_mask>0])
#             print("[Check loss] criterion: {}, loss manual calc: {}".format(loss_train_nonull_validate_nonull, loss_calc_out))
            
#             import pdb; pdb.set_trace()
            # 对全为valid shape的矩阵，预先设置一些位点为NULL，+seq 送到模型进行预测
            if args.use_decode_loss:
                output,x_decode = model(seq_matrix, shape_matrix.reshape(-1, args.sequence_length, 1))
            else:
                if args.out_featuremap:
                    output,fmap = model(seq_matrix, shape_matrix.reshape(-1, args.sequence_length, 1))
                    # for c,i in enumerate(fmap):
                        # featuremap[c].append(fmap[c][:,:,:])
                    featuremap[0].append(fmap[0][:,:,:])
                    featuremap[1].append(fmap[1][:,:,:])
                    featuremap[2].append(fmap[2][:,:,:])
                    # featuremap[3].append(fmap[3][:,:])
                else:
                    output = model(seq_matrix, shape_matrix.reshape(-1, args.sequence_length, 1))
            if args.lstm_all_time: output = output[:,-1,:]
            prediction.append(output)
            
            # print("[Check shape] validate output: {}".format(output.shape))
            loss['train_hasnull_validate_nonull'] += criterion(output[shape_null_mask>0], shape_true_matrix[shape_null_mask>0])
            loss['train_hasnull_validate_hasnull'] += criterion(output, shape_true_matrix)
            loss['train_hasnull_validate_onlynull'] += criterion(output[shape_null_mask<1], shape_true_matrix[shape_null_mask<1])
            
            # DMSseq loss
            loss['DMSloss_all'] = criterion(output[shape_true_matrix_null_mask>0], shape_true_matrix[shape_true_matrix_null_mask>0])
            loss['DMSloss_maskonly'] = criterion(output[np.logical_and(shape_true_matrix_null_mask>0, shape_null_mask<1)], shape_true_matrix[np.logical_and(shape_true_matrix_null_mask>0, shape_null_mask<1)])
            
            if args.print_train_time:
                print("[Check shape] validate shape_matrix: {},shape_null_mask: {}, shape_true_matrix: {}, shape_true_matrix_null_mask: {}".format(shape_matrix.shape, shape_null_mask.shape, shape_true_matrix.shape, shape_true_matrix_null_mask.shape))
                if torch.all(torch.eq(shape_true_matrix[shape_true_matrix_null_mask>0], shape_true_matrix.view(shape_true_matrix.numel()))): 
                    print("=")
                    print(criterion(output[shape_true_matrix_null_mask>0],shape_true_matrix[shape_true_matrix_null_mask>0]))
                    print(criterion(output, shape_true_matrix))
                    print(loss["train_nonull_validate_nonull"], loss['train_hasnull_validate_hasnull'])
                else:
                    print('!=')
            # print(output[shape_null_mask<1], shape_true_matrix[shape_null_mask<1])
            # if torch.isnan(shape_true_matrix[shape_null_mask<1]).sum(): print(shape_true_matrix[shape_null_mask<1])
            # print("[Check value] output[shape_null_mask<1] and shape_true_matrix[shape_null_mask<1]",
            # output[shape_null_mask<1], shape_true_matrix[shape_null_mask<1])
            
            """
            random_null = util.random_mask(x=shape_matrix.shape[0],
                                           y=shape_matrix.shape[1],
                                           mask_num=int(np.product(shape_matrix.shape)*0.1))
            shape_matrix_add_random_null = shape_matrix.clone()
            shape_matrix_add_random_null[random_null==0] = -1
            shape_matrix_add_random_null = shape_matrix_add_random_null.float()
            print(type(seq_matrix),type(shape_matrix_add_random_null))
            output = model(seq_matrix, shape_matrix_add_random_null.reshape(-1, args.sequence_length, 1))
            random_null_loss += criterion(output[shape_matrix_add_random_null==0], shape_matrix[shape_matrix_add_random_null==0])
            """

    # 这里应该除以batch的大小，因为上面是每个batch的mean loss，所以是多除了，在main里面又乘回来了(sum(loss)/batches => sum(loss)/data * batch_size=sum(loss)/(data/batch_size)=sum(loss)/batches)
    # 对于test/validate每一个batch，最好计算sum loss，最后总的都相加，再除以data大小。
    loss["train_nonull_validate_nonull"] /= len(validate_loader.dataset) 
    loss['train_hasnull_validate_nonull'] /= len(validate_loader.dataset)
    loss['train_hasnull_validate_hasnull'] /= len(validate_loader.dataset)
    loss['train_hasnull_validate_onlynull'] /= len(validate_loader.dataset)
    
    loss['DMSloss_all'] /= len(validate_loader.dataset)
    loss['DMSloss_maskonly'] /= len(validate_loader.dataset)
    
    prediction_all = torch.cat(prediction).tolist()
    if save_prediction:
        with open(save_prediction, 'w') as SAVEFN:
            for i in prediction_all:
                SAVEFN.write(','.join(map(str,i))+'\n')

    print('\n[Validation monitor] Validation set loss:\n\t train_nonull_validate_nonull: {:.8f} \n\t train_hasnull_validate_nonull: {:.8f} \n\t train_hasnull_validate_hasnull: {:.8f} \n\t train_hasnull_validate_onlynull: {:.8f}'.format(loss["train_nonull_validate_nonull"], \
                                                                                                        loss['train_hasnull_validate_nonull'], \
                                                                                                        loss['train_hasnull_validate_hasnull'], \
                                                                                                        loss['train_hasnull_validate_onlynull']))
    
    if args.monitor_val_loss.startswith('DMS'):
        print(' DMSloss_all: {:.8f} \n\t DMSloss_maskonly: {:.8f}'.format(loss['DMSloss_all'], loss['DMSloss_maskonly']))
    
    if args.out_featuremap:
            return loss,prediction_all,featuremap
    return loss,prediction_all
