
import os
import sys
import numpy as np
import matplotlib.pyplot as plot
import cls_feature_class
import cls_data_generator
import parameters
import time
from time import gmtime, strftime
import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
plot.switch_backend('agg')
from IPython import embed
from cls_compute_seld_results import ComputeSELDResults, reshape_3Dto2D
from SELD_evaluation_metrics import distance_between_cartesian_coordinates
import seldnet_model 
from model import NGCCModel
from speechbrain.nnet.losses import PitWrapper 
from cst_former.CST_former_model import CST_former
from torchinfo import summary
from warmup_scheduler import GradualWarmupScheduler
from train_seldnet import *
import random

seed_everything(42)

class Inference:
    def __init__(self):
    
    def load_model(self): #모델 weight load
        return None 
    
    def feature_extraction(self): #seld_feat_label 추출 
        return None
    
    def inference(self):
        return None
    
    