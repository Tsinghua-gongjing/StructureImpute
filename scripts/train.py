from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import copy,time

import util

def train_LSTM(args, model, device, train_loader, optimizer, epoch):
    model.train()
    # criterion = nn.SmoothL1Loss()
    if args.train_loss_func == 'mse':
        criterion = nn.MSELoss(reduction='mean')
    elif args.train_loss_func == 'exp_shape':
        criterion = util.loss_shape_exp()
    elif args.train_loss_func == 'reg_focal_loss':
        criterion = util.reg_focal_loss()
    elif args.train_loss_func == 'smooth_l1_loss':
        criterion = util.smooth_l1_loss()
    else:
        pass
    before_train_time = time.time()
    for batch_idx, (seq_matrix, shape_matrix, shape_matrix_null_mask,shape_true_matrix,
                    shape_true_matrix_null_mask) in enumerate(train_loader):
        
        # check data in each batch
        # print(batch_idx, seq_matrix.shape, shape_matrix.shape, seq_matrix[0:2,:,:], shape_matrix[0:2,:])
        # continue
        
        batch_loaddata_start_time = time.time()
        seq_matrix = seq_matrix.float().reshape(-1, args.sequence_length, args.input_size).cuda()
        shape_matrix = shape_matrix.float().cuda()
        shape_true_matrix = shape_true_matrix.float().cuda()
        optimizer.zero_grad()
        batch_loaddata_end_time = time.time()
        # print("[Check shape] training:",batch_idx, seq_matrix.shape, shape_matrix.reshape(-1, sequence_length, 1).shape, shape_true_matrix.shape)
        # print(args)
        
        if args.train_type == 'trainNoNull_lossAll':
            # 测试：输入shape不带有NULL
            output = model(x=seq_matrix, y=shape_matrix.reshape(-1, args.sequence_length, 1))
            # 只对非null值计算loss
            loss = criterion(output[shape_matrix_null_mask>0], shape_matrix[shape_matrix_null_mask>0]) 
        
        if args.train_type == 'trainHasNull_lossAll':
            # 测试：输入shape带有NULL
            if args.use_decode_loss:
                output,x_decode = model(x=seq_matrix, y=shape_matrix.reshape(-1, args.sequence_length, 1)) 
            else:
                output = model(x=seq_matrix, y=shape_matrix.reshape(-1, args.sequence_length, 1)) 
            # print("[Check shape] training output:",output.shape)
            # 此时训练数据含有随机的NULL，所有的碱基都参与计算loss
            if args.lstm_all_time:
                # print('[0]',shape_true_matrix.shape, shape_matrix_null_mask.shape)
                shape_true_matrix = shape_true_matrix.view(shape_true_matrix.shape[0], -1, args.sequence_length)
                shape_matrix_null_mask = shape_matrix_null_mask.view(shape_matrix_null_mask.shape[0], -1, args.sequence_length)
                # print('[1]',shape_true_matrix.shape, shape_matrix_null_mask.shape)
                shape_true_matrix = shape_true_matrix.repeat(1, args.sequence_length, 1)
                shape_matrix_null_mask = shape_matrix_null_mask.repeat(1, args.sequence_length, 1)
                # print('[2]',shape_true_matrix.shape, shape_matrix_null_mask.shape, output.shape)
            if args.use_decode_loss:
                loss         = criterion(output, shape_true_matrix) + criterion(seq_matrix, x_decode)
            else:
                loss         = criterion(output, shape_true_matrix)
            loss_nonnull = criterion(output[shape_matrix_null_mask>0], shape_true_matrix[shape_matrix_null_mask>0])
            loss_null    = criterion(output[shape_matrix_null_mask<1], shape_true_matrix[shape_matrix_null_mask<1])
        
        # print(batch_idx,data.shape,output.shape,target.shape)
        # print(output[0])
        # print(target[0])
        
        if args.train_type == 'trainHasNull_lossNullOnly':
            # 测试：输入shape带有NULL
            output = model(x=seq_matrix, y=shape_matrix.reshape(-1, args.sequence_length, 1))
            # 此时训练数据含有随机的NULL，只让本身为NULL的碱基参与计算loss
            loss = criterion(output[shape_matrix_null_mask<1], shape_true_matrix[shape_matrix_null_mask<1])
            
        if args.train_type == 'trainHasNull_lossnull':
            # 测试：输入shape带有NULL
            output = model(x=seq_matrix, y=shape_matrix.reshape(-1, args.sequence_length, 1))
            # 考虑非NULL和NULL权重，给予不同的权重
            # 这个理设置weight=1时，和上面的trainHasNull_lossAll方式得到的结果不一样，因为nonnull和null的数据量比例是90%、10%
            loss = criterion(output[shape_matrix_null_mask>0],shape_true_matrix[shape_matrix_null_mask>0]) + \
                    args.train_loss_null_weight*criterion(output[shape_matrix_null_mask<1], shape_true_matrix[shape_matrix_null_mask<1])
                
        if args.train_type == 'DMSloss_all':
            # 对于DMS，shape_matrix_null_mask包含原本没测到的-1和mask的-1，
            # shape_true_matrix_null_mask包含原本没测到的-1
            # 对于所有测到的都计算loss，包括mask的及不mask的
            output = model(x=seq_matrix, y=shape_matrix.reshape(-1, args.sequence_length, 1))
            loss = criterion(output[shape_true_matrix_null_mask>0], shape_true_matrix[shape_true_matrix_null_mask>0])
        if args.train_type == 'DMSloss_maskonly':
            # 对于DMS，shape_matrix_null_mask包含原本没测到的-1和mask的-1，
            # shape_true_matrix_null_mask包含原本没测到的-1
            # 对于所有测到的都计算loss，但是仅包括mask的碱基
            output = model(x=seq_matrix, y=shape_matrix.reshape(-1, args.sequence_length, 1))
            loss = criterion(output[np.logical_and(shape_true_matrix_null_mask>0, shape_matrix_null_mask<1)], shape_true_matrix[np.logical_and(shape_true_matrix_null_mask>0, shape_matrix_null_mask<1)])
            
        batch_forward_time = time.time()
        loss.backward()
        batch_losscalc_time = time.time()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()
        batch_optimizer_time = time.time()
        if batch_idx % args.log_interval == 0:
            if args.train_type == 'trainHasNull_lossAll':
                print('[Train monitor] Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(seq_matrix), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
            else:
                print('[Train monitor] Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(seq_matrix), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
        if args.print_train_time:
            util.timer(start=batch_loaddata_start_time, end=batch_optimizer_time, description='Batch training time')
            util.timer(start=batch_loaddata_start_time, end=batch_loaddata_end_time, description='Load data time')
            util.timer(start=batch_loaddata_end_time, end=batch_forward_time, description='Model time')
            util.timer(start=batch_forward_time, end=batch_losscalc_time, description='Loss calc time')
            util.timer(start=batch_losscalc_time, end=batch_optimizer_time, description='Optimizer calc time')
                
    if args.train_type == 'trainHasNull_lossAll':
        return loss.item(),loss_nonnull.item(),loss_null.item()
    else:
        return loss.item()