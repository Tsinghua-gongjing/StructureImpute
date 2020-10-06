from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import copy

import util

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from .AllFusionNetAdd import *
from .AllFusionNetDE import *
from .AllFusionNetMultiplyDEChanelFeatureAddEDloss import *
from .AllFusionNetMultiplyDEChanelFeature import *
from .AllFusionNetMultiplyDE import *
from .AllFusionNetMultiply import *
from .AllFusionNetMultiply_test import *
from .AllFusionNetMultiply_SRU import *
from .AllFusionNetMultiplyRes2 import *
from .AllFusionNetMultiplyAttention import *
from .seqOnly import *
from .seqResLSTMFusionX import *
from .shapeOnly import *
from .AllFusionNetFCThenCombine import *
from .AllFusionNet2FCThenCombine import *
from .AllFusionNet2FCThenCombineFC import *
from .AllFusionNetMultiplySeqsize6 import *
from .AllFusionNetMultiplyAlltime import *

def model_entry(name, args):
    return globals()[name](args)