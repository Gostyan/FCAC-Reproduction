import os
import os.path as osp
import re
import json
import time
import h5py
from matplotlib.font_manager import json_dump
import numpy as np
import random
import librosa
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchaudio
from torchvision import transforms
import pandas as pd
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation


class NDS(Dataset):

    def __init__(self, root='./', phase='train', 
                 index_path=None, index=None, k=5, base_sess=None, data_type='audio', args=None):
        self.root = os.path.expanduser(root)
        self.root = root
        self.data_type = data_type
        self.phase = phase
        
        # 加载NSynth-100数据集的CSV文件
        self.all_train_df = pd.read_csv(os.path.join(root, "nsynth-100-fs_train.csv"))
        self.all_val_df = pd.read_csv(os.path.join(root, "nsynth-100-fs_val.csv"))
        self.all_test_df = pd.read_csv(os.path.join(root, "nsynth-100-fs_test.csv"))
        
        # 加载标签映射
        with open(os.path.join(root, "nsynth-100-fs_vocab.json"), 'r') as f:
            self.label_mapping = json.load(f)

        # 为每个数据添加数字标签
        self.all_train_df['label'] = self.all_train_df['instrument'].map(self.label_mapping)
        self.all_val_df['label'] = self.all_val_df['instrument'].map(self.label_mapping)
        self.all_test_df['label'] = self.all_test_df['instrument'].map(self.label_mapping)

        if phase == 'train':
            if base_sess:
                self.data, self.targets = self.SelectfromClasses(self.all_train_df, index, per_num=None)
            else:
                self.data, self.targets = self.SelectfromClasses(self.all_train_df, index, per_num=None)
        elif phase == 'val':
            if base_sess:
                self.data, self.targets = self.SelectfromClasses(self.all_val_df, index, per_num=None)
            else:
                self.data, self.targets = self.SelectfromClasses(self.all_test_df, index, per_num=k)
        elif phase =='test':
            self.data, self.targets = self.SelectfromClasses(self.all_test_df, index, per_num=None)
        

    def SelectfromClasses(self, df, index, per_num=None):
        data_tmp = []
        targets_tmp = []

        for i in index:
            ind_cl = np.where(i == df['label'])[0]
            
            k = 0
            for j in ind_cl:
                filename = df['filename'][j] + '.wav'  # NSynth数据集是wav格式
                # 根据audio_source确定子目录
                audio_source = df['audio_source'][j]
                if audio_source == 'nsynth-train':
                    subdir = 'nsynth-train'
                elif audio_source == 'nsynth-valid':
                    subdir = 'nsynth-valid'
                elif audio_source == 'nsynth-test':
                    subdir = 'nsynth-test'
                else:
                    subdir = 'nsynth-train'  # 默认
                
                path = os.path.join(self.root, 'The_NSynth_Dataset', subdir, 'audio', filename)
                
                # 检查
                if os.path.exists(path):
                    data_tmp.append(path)
                    targets_tmp.append(df['label'][j])
                    k += 1
                    if per_num is not None:
                        if k >= per_num:
                            break
              
        return data_tmp, targets_tmp

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, targets = self.data[i], self.targets[i]
        try:
            audio, sr = torchaudio.load(path)
            return audio.squeeze(0), targets
        except Exception as e:
            print(f"Error loading {path}: {e}")
            # 返回零音频和目标
            return torch.zeros(16000), targets  # 假设1秒的16kHz音频
