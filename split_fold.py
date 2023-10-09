from data__set import Data__Set, PairedDataSet
import os
from sklearn.model_selection import KFold
import torchvision.transforms as trns
import torch.utils.data as TD
import numpy as np
import torch
from torch.autograd import Variable
import pandas as pd
import matplotlib.pyplot as plt
def get_nested_sub_file(path): 
    img_list=[]
    for f in path:
        path_list=os.listdir(f)
        for n in path_list:
            img_list.append(os.path.join(f,n))
    return img_list

if __name__ == "__main__":

    #check date
    '''date_path='src/FDG_DATE.xlsx'
    metadata_path='src/FDG_metadata.xls'
    
    date_df = pd.read_excel(date_path)
    metadata_df = pd.read_excel(metadata_path)

    date_i=0

    df_cn = pd.DataFrame()
    df_mci = pd.DataFrame()
    df_ad = pd.DataFrame()
    for i in range(0,len(metadata_df)):
        print(metadata_df.iloc[i]['ID'], metadata_df.iloc[i]['StudyDate'])

        meta_id=metadata_df.iloc[i]['ID']
        try:
            meta_year=metadata_df.iloc[i]['StudyDate'].split('/')[0]
            meta_month=metadata_df.iloc[i]['StudyDate'].split('/')[1]
            meta_date=metadata_df.iloc[i]['StudyDate'].split('/')[2]
        except AttributeError:
            meta_year=metadata_df.iloc[i]['StudyDate'].year
            meta_month=metadata_df.iloc[i]['StudyDate'].month
            meta_date=metadata_df.iloc[i]['StudyDate'].date
        print( meta_year,meta_month,meta_date)

        print("=============")

        date_id=date_df.iloc[date_i]['ID']
        print(date_df.iloc[date_i]['ID'],date_df.iloc[date_i]['StudyDate'])
        try:
            
            #print(date_df.iloc[date_i][1].split('.'))
            date_year=date_df.iloc[date_i]['StudyDate'].split('.')[0]
            date_month=date_df.iloc[date_i]['StudyDate'].split('.')[1]
            date_d=date_df.iloc[date_i]['StudyDate'].split('.')[2]
        except AttributeError:

            date_year=date_df.iloc[date_i]['StudyDate'].strftime("%Y")
            date_month=date_df.iloc[date_i]['StudyDate'].strftime("%m")
            date_d=date_df.iloc[date_i]['StudyDate'].strftime("%d")
        print( date_year,date_month,date_d)
        
        if meta_id==date_id:
            
            if str(date_year).find(str(meta_year))!=-1 and str(date_month).find(str(meta_month))!=-1:# and str(date_d).find(str(meta_date))!=-1:
                date_i=date_i+1
                print("!!!!!!!!!!!!")

                if metadata_df.iloc[i]["Group"]=='CN':
                    df_cn=df_cn.append(metadata_df.iloc[i])
                elif metadata_df.iloc[i]["Group"]=='MCI':
                    df_mci=df_mci.append(metadata_df.iloc[i])
                elif metadata_df.iloc[i]["Group"]=='AD':
                    df_ad=df_ad.append(metadata_df.iloc[i])

                
    
        print(i,date_i)    
        print("--------------------------------------------------")
        #if metadata_df.iloc[i]['StudyDate'] and (valid_df.iloc[i].sum()-i)==0:
            
    print("----------------CN----------------")
    print(df_cn)

    print("----------------MCI----------------")
    print(df_mci)
    print("----------------AD----------------")
    print(df_ad)'''
    
    

    FDG_root='src/PET_112'
    FDG_dataset_list=os.listdir(FDG_root)
    for n in range(len(FDG_dataset_list)):
        FDG_dataset_list[n]=os.path.join(FDG_root,FDG_dataset_list[n])     

    MRI_root='src/MRI_112'
    MRI_dataset_list=os.listdir(MRI_root)
    for n in range(len(MRI_dataset_list)):
        MRI_dataset_list[n]=os.path.join(MRI_root,MRI_dataset_list[n])     
        
    #img0 = get_nested_sub_file(FDG_dataset_list)
    #img1 = get_nested_sub_file(MRI_dataset_list)
    img0 = np.asarray(FDG_dataset_list).reshape(-1,1)
    img1 = np.asarray(MRI_dataset_list).reshape(-1,1)
    
    print(img0.shape)

    test_fold_rotation=np.array([(0), (1), (2), (3), (4)])
    valid_fold_rotation=np.array([(1), (2), (3), (4), (0)])
    train_fold_rotation=np.array([(2,3,4), (0,3,4), (0,1,4), (0,1,2), (1,2,3)])

    for fold in range(5):
        
        print("=================FOLD{}======================".format(str(fold)))
        train_ids=[]
        valid_ids=[]
        test_ids=[]
        for j in train_fold_rotation[fold]:
            #print(j)
            train_ids.append(np.arange(20*j,20*(j+1)))
            
        
        valid_ids=np.arange(20*valid_fold_rotation[fold],20*(valid_fold_rotation[fold]+1))
        test_ids=np.arange(20*test_fold_rotation[fold],20*(test_fold_rotation[fold]+1))
        
        train_ids=np.array(train_ids).reshape(1,-1)
        valid_ids=np.array(valid_ids).reshape(1,-1)
        test_ids=np.array(test_ids).reshape(1,-1)
        print(train_ids.shape)
        print("train ids",train_ids)
        print("valid ids",valid_ids)
        print("test ids",test_ids)
        
        if not os.path.exists('src/fold/fold{}'.format(fold)):
            os.makedirs('src/fold/fold{}'.format(fold))
            
        path = 'src/fold/fold{}/valid_PET.txt'.format(fold)
   
        with open(path, 'w') as f:
            print("valid PET")
            for i in valid_ids[0]:
                filenames=os.listdir(img0[i][0])
                print(i,img0[i][0])
                for filename in filenames:
                    filename = os.path.join(img0[i][0],str(filename)[2:])#
                    
                    f.write(filename+'\n')
            
        path = 'src/fold/fold{}/train_PET.txt'.format(fold)
        with open(path, 'w') as f:
            print("train PET")
            for i in train_ids[0]:
                filenames=os.listdir(img0[i][0])
                print(i,img0[i][0])
                for filename in filenames:
                    filename = os.path.join(img0[i][0],str(filename)[2:])#
                    f.write(filename+'\n')
                    
        path = 'src/fold/fold{}/test_PET.txt'.format(fold)
        with open(path, 'w') as f:
            print("test PET")
            for i in test_ids[0]:
                filenames=os.listdir(img0[i][0])
                print(i,img0[i][0])
                for filename in filenames:
                    filename = os.path.join(img0[i][0],str(filename)[2:])#
                    f.write(filename+'\n')
            
        path = 'src/fold/fold{}/valid_MRI.txt'.format(fold)
        with open(path, 'w') as f:
            print("valid MRI")
            for i in valid_ids[0]:
                filenames=os.listdir(img0[i][0])
                print(i,img0[i][0].replace("PET","MRI"))
                #filenames = filenames.replace("FDG","MRI")
                for filename in filenames:
                    filename = os.path.join(img0[i][0].replace("PET","MRI"),str(filename)[2:])#
                    f.write(filename+'\n')
            
        path = 'src/fold/fold{}/train_MRI.txt'.format(fold)
        with open(path, 'w') as f:
            print("train MRI")
            for i in train_ids[0]:
                filenames=os.listdir(img0[i][0])
                print(i,img0[i][0].replace("PET","MRI"))
                for filename in filenames:
                    filename = os.path.join(img0[i][0].replace("PET","MRI"),str(filename)[2:])#
                    f.write(filename+'\n')
            
        path = 'src/fold/fold{}/test_MRI.txt'.format(fold)
        with open(path, 'w') as f:
            print("test MRI")
            for i in test_ids[0]:
                filenames=os.listdir(img0[i][0])
                print(i,img0[i][0].replace("PET","MRI"))
                for filename in filenames:
                    filename = os.path.join(img0[i][0].replace("PET","MRI"),str(filename)[2:])#
                    f.write(filename+'\n')
        #print(range(20*i,20*(i+1)))
        '''img0_fold=img0[20*i:20*(i+1)]
        print(img0_fold)
        print(i,"------------------")'''
        
        #train_ids

    #print()

    #dataset = Data__Set(root0=FDG_dataset_list,root1=MRI_dataset_list,root2=MRI_dataset_list,transform=trns.Compose([trns.ToTensor()]))
    '''kfold = KFold(n_splits=5, shuffle=True)
    for fold, (train_ids, valid_ids) in enumerate(kfold.split(np.random.randn(len(img0),1))):
        print("------------------------------------------", "fold ", fold, "------------------------------------------")
        
        print(train_ids[0:5])
        print(img0[train_ids[0:5]])
        print(img1[train_ids[0:5]])

        print(valid_ids[0:5])
        print(img0[valid_ids[0:5]])
        print(img1[valid_ids[0:5]])
        
        
        if not os.path.exists('src/fold/fold{}'.format(fold)):
            os.makedirs('src/fold/fold{}'.format(fold))
            
        path = 'src/fold/fold{}/valid_PET.txt'.format(fold)
   
        with open(path, 'w') as f:
            for i in valid_ids:
                filenames=os.listdir(img0[i][0])
                for filename in filenames:
                    filename = os.path.join(img0[i][0],str(filename)[2:])#
                    
                    f.write(filename+'\n')
            
        path = 'src/fold/fold{}/train_PET.txt'.format(fold)
        with open(path, 'w') as f:
            for i in train_ids:
                filenames=os.listdir(img0[i][0])
                for filename in filenames:
                    filename = os.path.join(img0[i][0],str(filename)[2:])#
                    f.write(filename+'\n')
            
        path = 'src/fold/fold{}/valid_MRI.txt'.format(fold)
        with open(path, 'w') as f:
            for i in valid_ids:
                filenames=os.listdir(img0[i][0])
                #filenames = filenames.replace("FDG","MRI")
                for filename in filenames:
                    filename = os.path.join(img0[i][0].replace("PET","MRI"),str(filename)[2:])#
                    f.write(filename+'\n')
            
        path = 'src/fold/fold{}/train_MRI.txt'.format(fold)
        with open(path, 'w') as f:
            for i in train_ids:
                filenames=os.listdir(img0[i][0])
                for filename in filenames:
                    filename = os.path.join(img0[i][0].replace("PET","MRI"),str(filename)[2:])#
                    f.write(filename+'\n')
'''
        
        #print(dataset[valid_ids[0]][1])
        #train_subsampler = TD.SubsetRandomSampler(train_ids)
        #valid_subsampler = TD.SubsetRandomSampler(valid_ids)
        
    
    '''MRI_dataset_list=[]
    FDG_dataset_list=[] 

    f = open('src/fold/fold0/valid_MRI.txt', 'r')
    for line in f.readlines():
        MRI_dataset_list.append(str(line)[:-2])
        #print(str(line[:-1]))
    f = open('src/fold/fold0/valid_PET.txt', 'r')
    for line in f.readlines():
        FDG_dataset_list.append(str(line)[:-2])
        
    dataset = PairedDataSet(root0=FDG_dataset_list,root1=MRI_dataset_list,transform=trns.Compose([trns.ToTensor()]))
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=16)
    for iteration, data in enumerate(trainloader, 1):
        #print(trainloader[17703])
        batch=data[0]
        batch_path=data[1]
        print(len(trainloader),iteration)#,batch_path)
        FDG_1c = Variable(batch[:,0,0,:,:],requires_grad=True)
        FDG_1c = FDG_1c[:,None,:,:]'''
        
        #print(FDG_1c.shape)
        
  
    #print(MRI_dataset_list)    
    '''text = f.read()
    text = np.asarray(text).reshape(-1,1)
    print(text.shape)
    f.close'''