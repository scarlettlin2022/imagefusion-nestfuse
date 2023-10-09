
#unedited
from hashlib import new
from PIL import Image
#from scipy.io import loadmat
from torch.utils.data.dataset import Dataset
import torchvision.datasets as dsets
import os
import torchvision.transforms as trns
import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.utils.data as data
from sklearn.model_selection import train_test_split


class Data__Set(Dataset):
    def __init__(self, root0, root1, root2, transform):

        self.transform = transform
        self.root0=root0
        self.root1=root1
        self.root2=root2
        
        self.imgs0=get_nested_sub_file(root0)
        self.imgs1=get_nested_sub_file(root1)
        self.imgs2=get_nested_sub_file(root2)


    def __getitem__(self, index):

        imgpath0 = self.imgs0[index]
        img0 = np.load(imgpath0)
        img0=img0.astype(np.float32)
        #print('img0.shape',img0.shape)
        #print('np.max(img0)',np.max(img0))
        #print('np.min(img0)',np.min(img0))
        '''if self.transform is not None:
            img0 = self.transform(img0)'''
        #print('img0.shape after transform',img0.shape)    
        imgpath1 = self.imgs1[index]
        img1 = np.load(imgpath1)
        img1= img1.astype(np.float32)
        
        imgpath2 = self.imgs2[index]
        img2 = np.load(imgpath2)
        img2= img2.astype(np.float32)
        
        #print(imgpath0)
        
        #print('img1.shape',img1.shape)
        #print('np.max(img1)',np.max(img1))
        #print('np.min(img1)',np.min(img1))
        '''if self.transform is not None:
            img1 = self.transform(img1)'''
        #print('img1.shape after transform',img1.shape)
        
        o=np.concatenate([img0[None,:,:,:],img1[None,:,:,:], img2[None,:,:,:]],axis=0)
        #print('o.shape',o.shape)
        #print(o)
        return o, imgpath0

    def __len__(self):
        return len(self.imgs0)
    
    def __path__(self):
        return self.imgs0,self.imgs1,self.imgs2
    
def get_sub_file(path): 
    path_list=os.listdir(path)
    for n in range(len(path_list)):
        path_list[n]=os.path.join(path,path_list[n]) 
    return path_list

def get_nested_sub_file(path): 
    img_list=[]
    for f in path:
        path_list=os.listdir(f)
        #print('path_list',path_list)
        for n in path_list:
            #img_list+=os.path.join(f,n)
            img_list.append(os.path.join(f,n))
            #print(os.path.join(f,n))
        #print(len(img_list))
    #print('img_list',img_list)
    return img_list
     
     
class PairedDataSet(Dataset):
    def __init__(self, root0, root1, transform):

        self.transform = transform
        self.root0=root0
        self.root1=root1



    def __getitem__(self, index):

        imgpath0 = self.root0[index]
        img0 = np.load(imgpath0)
        img0=img0.astype(np.float32)
         
        imgpath1 = self.root1[index]
        img1 = np.load(imgpath1)
        img1= img1.astype(np.float32)
        
        #print(index, imgpath0, imgpath1)
        #print(self.root0[17704])
        
        o=np.concatenate([img0[None,:,:,:],img1[None,:,:,:]],axis=0)
        #print('o.shape',o.shape)
        #print(o)
        return o, imgpath0

    def __len__(self):
        return len(self.root0)
    
class TriDataSet(Dataset):
    def __init__(self, root0, root1, root2, transform):

        self.transform = transform
        self.root0=root0
        self.root1=root1
        self.root2=root2



    def __getitem__(self, index):

        imgpath0 = self.root0[index]
        img0 = np.load(imgpath0)
        img0=img0.astype(np.float32)
         
        imgpath1 = self.root1[index]
        img1 = np.load(imgpath1)
        img1= img1.astype(np.float32)
        
        imgpath2 = self.root2[index]
        img2 = np.load(imgpath2)
        img2= img2.astype(np.float32)
        
        #print(index, imgpath0, imgpath1)
        #print(self.root0[17704])
        
        o=np.concatenate([img0[None,:,:,:],img1[None,:,:,:],img2[None,:,:,:]],axis=0)
        #print('o.shape',o.shape)
        #print(o)
        return o, imgpath0

    def __len__(self):
        return len(self.root0) 


if __name__ == '__main__':
    '''FDG_root=r'src/PET_112'
    FDG_dataset_list=os.listdir(FDG_root)
    #FDG_dataset_list=os.path.join(FDG_root,FDG_dataset_list)
    for n in range(len(FDG_dataset_list)):
        FDG_dataset_list[n]=os.path.join(FDG_root,FDG_dataset_list[n])     
    #print(FDG_dataset_list)
    
    MRI_root='src/MRI_112_slice_norm'
    MRI_dataset_list=os.listdir(MRI_root)
    for n in range(len(MRI_dataset_list)):
        MRI_dataset_list[n]=os.path.join(MRI_root,MRI_dataset_list[n])     
    #print(MRI_dataset_list)
    
    FDG_train, FDG_eval, MRI_train, MRI_eval = train_test_split(FDG_dataset_list, MRI_dataset_list, test_size=0.2, random_state=42)
    
    print('FDG_train',len(FDG_train))
    print('MRI_train',len(MRI_train))
    
    FDG_valid, FDG_test, MRI_valid, MRI_test = train_test_split(FDG_eval, MRI_eval, test_size=0.5, random_state=42)
    
    print('FDG_valid',len(FDG_valid))
    print('MRI_test',len(MRI_valid))
    print('FDG_test',len(FDG_test))
    print('MRI_test',len(MRI_test))
    
    MRI_train_inverted=[element.replace('MRI_112_slice_norm', 'MRI_112_inverted') for element in MRI_train]
    MRI_valid_inverted=[element.replace('MRI_112_slice_norm', 'MRI_112_inverted') for element in MRI_valid]
    MRI_test_inverted=[element.replace('MRI_112_slice_norm', 'MRI_112_inverted') for element in MRI_test]

    train_transform = trns.Compose([
            trns.ToTensor()
        ])
    train_set = PairedDataSet(root0=FDG_train[0:2],root1=MRI_train[0:2], root2=MRI_train_inverted[0:2], transform=train_transform)
    print(train_set[0])'''


    MRI_valid=[]
    FDG_valid=[] 
    MRI_inverted_valid=[]
    

    f_name='src/fold/fold0/valid_MRI.txt'
    f_name=f_name.replace('fold/', 'fold_112_slice_norm/')
    f = open(f_name, 'r')  
    for line in f.readlines():
        MRI_valid.append(str(line)[:-2])
        print(str(line)[:-2])
        
    f_name='src/fold/fold0/valid_PET.txt'
    f_name=f_name.replace('fold/', 'fold_112_slice_norm/')
    f = open(f_name, 'r')
    for line in f.readlines():
        FDG_valid.append(str(line)[:-2])
        
    f_name='src/fold/fold0/valid_MRI.txt'
    f = open(f_name, 'r')  
    for line in f.readlines():
        MRI_inverted_valid.append(str(line)[:-2].replace('MRI_112/', 'MRI_112_inverted/'))
        print(str(line)[:-2].replace('MRI_112/', 'MRI_112_inverted/'))
        #print(str(line[:-1]))
    
    #train_dataset = PairedDataSet(root0=FDG_train,root1=MRI_train,transform=trns.Compose([trns.ToTensor()]))
    valid_dataset = TriDataSet(root0=FDG_valid,root1=MRI_valid,root2=MRI_inverted_valid,transform=trns.Compose([trns.ToTensor()]))
    print(valid_dataset[0][1])
        
    
    
    