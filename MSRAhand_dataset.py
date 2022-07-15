# author: LiyuanQi

from torch.utils.data import Dataset
import pickle
import os
import numpy as np
# from natsort import natsorted, ns
# from open3d import io as o3dio
# from open3d import geometry as o3dg
# from open3d import utility as o3du
# from open3d import t as o3dt
osp = os.path
#from datapreprocess import data_process
import torch

class MSRAhand(Dataset):
    def __init__(self,n_sample,task):
        super(MSRAhand,self).__init__()
        gesture_list = ['1','2','3','4','5','6','7','8','9','I','IP','L','MP','RP','T','TIP','Y']
        if n_sample == 1024:
            self.datadir = '/home/liyuan/HandEstimation/dataset/MSRAhand_data_1024'
        if n_sample == 2048:
            self.datadir = '/home/liyuan/HandEstimation/dataset/MSRAhand_data_2048'
        if task == 'train':
            p_list = ['P0','P1','P2','P3','P4','P5','P6','P7']
        if task == 'test':
            p_list = ['P8']
        if task == 'all':
            p_list = ['P0','P1','P2','P3','P4','P5','P6','P7','P8']
        self.p_len = {}
        self.g_len = {}
        for p_num in p_list:
            curr_g_len = {}
            g_len = {}
            for g_num in gesture_list:
                
                pcd_folder = osp.join(self.datadir,p_num,g_num)
                curr_g_len[g_num] = len(os.listdir(pcd_folder))
            self.p_len[p_num] = sum(curr_g_len.values())
            self.g_len[p_num] = curr_g_len
        self.len = sum(self.p_len.values())

    
    def reindex(self,index):
        assert index <= self.len
        [*p_nums],[*lens_p] = zip(*self.p_len.items())
        
        i = 0
        while index - sum(lens_p[:i+1]) >= 0:
            i = i+1
        index = index - sum(lens_p[:i])
        p_num = p_nums[i]
        
        [*g_nums],[*lens_g] = zip(*self.g_len[p_num].items())
        j = 0
        while index - sum(lens_g[:j+1]) >= 0:
            j = j+1
        index = index - sum(lens_g[:j])
        g_num = g_nums[j]
        
        return p_num,g_num,index
        
    
    def __len__(self):
        return self.len
    
    def __getitem__(self,index):
        p,g,index = self.reindex(index = index)
        point_dir = osp.join(self.datadir,p,g)
        point_file = os.listdir(point_dir)
        point_file = natsorted(point_file, key=lambda y: y.lower())
        point_num = point_file[index]
        pcd = o3dio.read_point_cloud(osp.join(point_dir,point_num))
        label_id = point_num.split('.')[0]
        
        label_dir = osp.join(self.datadir,p,'label.pkl')
        with open(label_dir,'rb') as fl:
            label_ = pickle.load(fl)
        labels = np.array(label_[g][int(label_id)])/1000
        points = np.asarray(pcd.points)/1000
        current_points = torch.from_numpy(points.copy()).float()
        current_labels = torch.flatten(torch.from_numpy(labels.copy()).float())
        return current_points,current_labels

class MSRAhand_n(Dataset):
    def __init__(self,task):
        super(MSRAhand_n,self).__init__()
        if task == 'train':
            dir = '/xavierssd250/Pointnet2.PyTorch/data_ver2/train'
        if task == 'test':
            dir = '/xavierssd250/Pointnet2.PyTorch/data_ver2/test'
        
        datadir = osp.join(dir,'data.pkl')
        labeldir = osp.join(dir,'label.pkl')
        with open(datadir,'rb') as fd:
            self.data = pickle.load(fd)
        with open(labeldir,'rb') as fl:
            self.label = pickle.load(fl)
        self.len = len(self.data)
    def __len__(self):
        return self.len
    def __getitem__(self,index):
        return self.data[index],self.label[index]
        
if __name__ == '__main__':
    from torch.utils.data import DataLoader
    data = MSRAhand_n()
    dloader = DataLoader(data, batch_size = 1, shuffle=True)
    inputs, labels = next(iter(dloader))
    print(inputs[0])