from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
from nested_dict import nested_dict
# import util

class Engine(object):
    def __init__(self, model, device, args):
        self.model  = model
        self.device = device
        self.args   = args

    def train(self, train_loader, optimizer, epoch):
        device = self.device
        args = self.args
        self.model.train()
        criterion = nn.MSELoss(reduction='mean')
        for batch_idx, (seq_matrix, shape_matrix, shape_matrix_null_mask,shape_true_matrix,
                        shape_true_matrix_null_mask) in enumerate(train_loader):
            
            seq_matrix = seq_matrix.float().reshape(-1, args.sequence_length, args.input_size).to(device)
            shape_matrix = shape_matrix.float().to(device)
            shape_true_matrix = shape_true_matrix.float().to(device)
            optimizer.zero_grad()
            
            if args.train_type == 'trainNoNull_lossAll':
                output = self.model(x=seq_matrix, y=shape_matrix.reshape(-1, args.sequence_length, 1))
                mask = shape_matrix_null_mask>0
                loss = criterion(output[mask], shape_true_matrix[mask])

            if args.train_type == 'trainHasNull_lossAll':
                if args.use_decode_loss:
                    output,x_decode = self.model(x=seq_matrix, y=shape_matrix.reshape(-1, args.sequence_length, 1)) 
                else:
                    output = self.model(x=seq_matrix, y=shape_matrix.reshape(-1, args.sequence_length, 1)) 
                if args.lstm_all_time:
                    shape_true_matrix      = shape_true_matrix.view(shape_true_matrix.shape[0], -1, args.sequence_length)
                    shape_matrix_null_mask = shape_matrix_null_mask.view(shape_matrix_null_mask.shape[0], -1, args.sequence_length)
                    shape_true_matrix      = shape_true_matrix.repeat(1, args.sequence_length, 1)
                    shape_matrix_null_mask = shape_matrix_null_mask.repeat(1, args.sequence_length, 1)
                mask0 = shape_matrix_null_mask<1
                mask1 = shape_matrix_null_mask>0
                #if args.use_decode_loss:
                #    loss         = criterion(output, shape_true_matrix) + criterion(seq_matrix, x_decode)
                #else:
                #    loss         = criterion(output, shape_true_matrix)
                loss_nonnull = criterion(output[mask1], shape_true_matrix[mask1])
                loss_null    = criterion(output[mask0], shape_true_matrix[mask0])
                loss = loss_nonnull + args.train_loss_null_weight*loss_null
            
            if args.train_type == 'trainHasNull_lossNullOnly':
                output = self.model(x=seq_matrix, y=shape_matrix.reshape(-1, args.sequence_length, 1))
                mask = shape_matrix_null_mask<1
                loss = criterion(output[mask], shape_true_matrix[mask])
                
            if args.train_type == 'trainHasNull_lossnull':
                output = self.model(x=seq_matrix, y=shape_matrix.reshape(-1, args.sequence_length, 1))
                # 考虑非NULL和NULL权重，给予不同的权重
                # 这个理设置weight=1时，和上面的trainHasNull_lossAll方式得到的结果不一样，因为nonnull和null的数据量比例是90%、10%
                mask0 = shape_matrix_null_mask<1
                mask1 = shape_matrix_null_mask>0
                loss_nonnull = criterion(output[mask1], shape_true_matrix[mask1])
                loss_null    = criterion(output[mask0], shape_true_matrix[mask0])
                loss = loss_nonnull + args.train_loss_null_weight*loss_null
                    
            if args.train_type == 'DMSloss_all':
                # 对于DMS，shape_matrix_null_mask包含原本没测到的-1和mask的-1，
                # shape_true_matrix_null_mask包含原本没测到的-1
                # 对于所有测到的都计算loss，包括mask的及不mask的
                output = self.model(x=seq_matrix, y=shape_matrix.reshape(-1, args.sequence_length, 1))
                mask = shape_true_matrix_null_mask>0
                loss = criterion(output[mask], shape_true_matrix[mask])
            if args.train_type == 'DMSloss_maskonly':
                # 对于DMS，shape_matrix_null_mask包含原本没测到的-1和mask的-1，
                # shape_true_matrix_null_mask包含原本没测到的-1
                # 对于所有测到的都计算loss，但是仅包括mask的碱基
                output = self.model(x=seq_matrix, y=shape_matrix.reshape(-1, args.sequence_length, 1))
                mask = np.logical_and((shape_true_matrix_null_mask>0).numpy(), (shape_matrix_null_mask<1).numpy())#.bool()
                loss = criterion(output[mask], shape_true_matrix[mask])
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip_norm, norm_type=2)
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                if args.train_type == 'trainHasNull_lossAll':
                    print('[Train monitor] Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(seq_matrix), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()))
                else:
                    print('[Train monitor] Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(seq_matrix), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()))
                    
        if args.train_type == 'trainHasNull_lossAll':
            return loss.item(),loss_nonnull.item(),loss_null.item()
        else:
            return loss.item()



    def validate(self, validate_loader, save_prediction=None):
        device = self.device
        args = self.args
        self.model.eval()

        # validate_loss = 0
        # random_null_loss = 0
        loss = nested_dict(1, int)
        criterion = nn.MSELoss(reduction='mean')
        prediction = []
        featuremap = [[],[],[]]
        i_batch = 0
        with torch.no_grad():
            for seq_matrix, shape_matrix, shape_null_mask,shape_true_matrix,shape_true_matrix_null_mask in validate_loader:
                i_batch += 1
                
                seq_matrix = seq_matrix.float().reshape(-1, args.sequence_length, args.input_size).to(device)
                shape_matrix = shape_matrix.float().to(device)
                shape_true_matrix = shape_true_matrix.float().to(device)
                
                # 对全为valid shape的矩阵，+seq 送到模型进行预测
                if args.use_decode_loss:
                    output,x_decode = self.model(seq_matrix, shape_true_matrix.reshape(-1, args.sequence_length, 1))
                else:
                    if args.out_featuremap:
                        output,fmap = self.model(seq_matrix, shape_matrix.reshape(-1, args.sequence_length, 1))
                    else:
                        output = self.model(seq_matrix, shape_true_matrix.reshape(-1, args.sequence_length, 1))
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
                mask = shape_true_matrix_null_mask>0
                loss_train_nonull_validate_nonull = criterion(output[mask], shape_true_matrix[mask]) 
                loss["train_nonull_validate_nonull"] += loss_train_nonull_validate_nonull
                # loss_calc = nn.MSELoss()
                # loss_calc_out = loss_calc(output[shape_true_matrix_null_mask>0],shape_true_matrix[shape_true_matrix_null_mask>0])
                
                # 对全为valid shape的矩阵，预先设置一些位点为NULL，+seq 送到模型进行预测
                if args.use_decode_loss:
                    output,x_decode = self.model(seq_matrix, shape_matrix.reshape(-1, args.sequence_length, 1))
                else:
                    if args.out_featuremap:
                        output,fmap = self.model(seq_matrix, shape_matrix.reshape(-1, args.sequence_length, 1))
                        # for c,i in enumerate(fmap):
                            # featuremap[c].append(fmap[c][:,:,:])
                        featuremap[0].append(fmap[0][:,:,:])
                        featuremap[1].append(fmap[1][:,:,:])
                        featuremap[2].append(fmap[2][:,:,:])
                        # featuremap[3].append(fmap[3][:,:])
                    else:
                        output = self.model(seq_matrix, shape_matrix.reshape(-1, args.sequence_length, 1))
                if args.lstm_all_time: 
                    output = output[:,-1,:]
                prediction.append(output)
                
                mask0 = shape_null_mask<1
                mask1 = shape_null_mask>0
                loss['train_hasnull_validate_nonull']   += criterion(output[mask1], shape_true_matrix[mask1])
                loss['train_hasnull_validate_hasnull']  += criterion(output, shape_true_matrix)
                loss['train_hasnull_validate_onlynull'] += criterion(output[mask0], shape_true_matrix[mask0])
                
                # DMSseq loss
                mask = shape_true_matrix_null_mask>0
                loss['DMSloss_all'] = criterion(output[mask], shape_true_matrix[mask])
                # import pdb;pdb.set_trace()
                mask = np.logical_and((shape_true_matrix_null_mask>0).numpy(), (shape_null_mask<1).numpy())#.bool()
                loss['DMSloss_maskonly'] = criterion(output[mask], shape_true_matrix[mask])
                
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

        loss["train_nonull_validate_nonull"]    /= i_batch
        loss['train_hasnull_validate_nonull']   /= i_batch
        loss['train_hasnull_validate_hasnull']  /= i_batch
        loss['train_hasnull_validate_onlynull'] /= i_batch
        
        loss['DMSloss_all']      /= i_batch
        loss['DMSloss_maskonly'] /= i_batch
        
        prediction_all = torch.cat(prediction).tolist()
        if save_prediction:
            with open(save_prediction, 'w') as SAVEFN:
                for i in prediction_all:
                    SAVEFN.write(','.join(map(str,i))+'\n')

    
        log_line = '\n[Validation monitor] Validation set loss:\n\t \
            train_nonull_validate_nonull:    {:.8f} \n\t \
            train_hasnull_validate_nonull:   {:.8f} \n\t \
            train_hasnull_validate_hasnull:  {:.8f} \n\t \
            train_hasnull_validate_onlynull: {:.8f}'.format(
                loss["train_nonull_validate_nonull"], \
                loss['train_hasnull_validate_nonull'], \
                loss['train_hasnull_validate_hasnull'], \
                loss['train_hasnull_validate_onlynull']
            )
        print(log_line)
        
        if args.monitor_val_loss.startswith('DMS'):
            print(' DMSloss_all: {:.8f} \n\t DMSloss_maskonly: {:.8f}'.format(
                loss['DMSloss_all'], 
                loss['DMSloss_maskonly']
                )
            )
        
        if args.out_featuremap:
                return loss,prediction_all,featuremap
        return loss,prediction_all
