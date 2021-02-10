import numpy as np
import torch
from torch.utils.data import Dataset


class StructureDataset(Dataset):
    def __init__(self, fragment):
        #, fragment_len, dataset, input_size, max_null_pct):
        self.fragment = fragment
        # self.fragment_len = fragment_len
        # self.dataset = dataset
        # self.feature_size = input_size
        # self.max_null_pct = max_null_pct
        print("loading data")
        data = np.load(fragment+".npz", allow_pickle=True)
        self.seq_matrix = data['seq']
        self.shape_matrix  = data['struct']
        self.shape_true_matrix  = data['struct_true']
        # self.seq_matrix,self.shape_matrix,self.shape_true_matrix = util.fragment_to_format_data(
        #     fragment=self.fragment, fragment_len=self.fragment_len, dataset=self.dataset, feature_size=self.feature_size, max_null_pct=self.max_null_pct)
        self.seq_matrix = torch.from_numpy(self.seq_matrix[0:])
        self.shape_matrix = torch.from_numpy(self.shape_matrix[0:])
        self.shape_true_matrix = torch.from_numpy(self.shape_true_matrix[0:])
        self.shape_matrix_null_mask = torch.ones_like(self.shape_matrix)
        self.shape_matrix_null_mask[self.shape_matrix<0] = 0 
        self.shape_true_matrix_null_mask = torch.ones_like(self.shape_true_matrix)
        self.shape_true_matrix_null_mask[self.shape_true_matrix<0] = 0
        self.len = self.seq_matrix.shape[0]
        print("[Check data len] {}: {}".format(self.fragment, self.len))
        
    def __getitem__(self, idx):
        return self.seq_matrix[idx], \
               self.shape_matrix[idx], \
               self.shape_matrix_null_mask[idx], \
               self.shape_true_matrix[idx], \
               self.shape_true_matrix_null_mask[idx]
    
    def __len__(self):
        return self.len