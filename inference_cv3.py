import argparse, os
from cProfile import label
import torch
import math, random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
#from torch.utils.data import DataLoader

#from dataset import DatasetFromHdf5
from torchvision import models
import torch.utils.model_zoo as model_zoo
from data__loader import Data__Loader
import time 
from sklearn.model_selection import train_test_split   
import matplotlib.pyplot as plt

from skimage.metrics import structural_similarity as ssim
import numpy as np
import math

from net_v1 import Net
from torch.optim import Adam
import pytorch_msssim
from args_fusion import args
import scipy.io as scio
import sys
from PIL import Image
import torchvision.transforms as trns
from data__set import Data__Set

# Training settings
parser = argparse.ArgumentParser(description="PyTorch SRResNet")

parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
parser.add_argument("--cuda", action="store_true", help="Use cuda?")
parser.add_argument("--threads", type=int, default=0, help="Number of threads for data loader to use, Default: 1")


parser.add_argument("--resume", default="checkpoint/gradientFalse_adaptiveFalse/best_loss_fold_0.pth", type=str, help="Path to checkpoint (default: none)")

parser.add_argument("--batchSize", type=int, default=16, help="training batch size")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--encoder", action="store_true",default=False ,  help="joint(0, Default)/seperate(1) encoder network")
parser.add_argument("--maskMRI", action="store_true",default=False ,  help="full MRI(0, Default)/grey matter only(1)")

parser.add_argument("--dtstCh1", default='src/FDG_test_delete', type=str, help="root folder: channel 1 of input dataset")
parser.add_argument("--dtstCh2", default='src/MRI_test_delete', type=str, help="root folder: channel 2 of input dataset")

parser.add_argument("--fusion", type=int, default=0,  help="spatial_channel(0, Default)/linear(1)/hydra(2)")
parser.add_argument("--heads", type=int, default=4, help="number of attention heads (recommend 1~16)")

global style_weights,content_weight,style_weight,layer_dict, output_folder
'''output_folder=str(time.localtime()[0])+"-"+str(time.localtime()[1])+"-"+str(time.localtime()[2])+"-"+str(time.localtime()[3])+str(time.localtime()[4])+str(time.localtime()[5])+'/'''
output_folder="output/"
#output_folder="vis"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
print('output folder: ', output_folder)

def load_data(path):
    print("=> Using the dataset from resume model")
    my_file = open(path, "r")
    data = my_file.read().split("\n")
    data1=data[:len(data)//2]
    data2=data[len(data)//2:]
    my_file.close()
    return data1, data2
def record_parser(opt):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    f = open(os.path.join(output_folder,"command.txt"), 'w')
    f.write(str(opt))
    f.close()


    
def load_model(path, deepsupervision):
	input_nc = 1
	output_nc = 1
	nb_filter = [64, 112, 160, 208, 256]

	nest_model = NestFuse_autoencoder(nb_filter, input_nc, output_nc, deepsupervision)
        
    
	nest_model.load_state_dict(torch.load(path))

	para = sum([np.prod(list(p.size())) for p in nest_model.parameters()])
	type_size = 4
	print('Model {} : params: {:4f}M'.format(nest_model._get_name(), para * type_size / 1000 / 1000))

	#nest_model.eval()
	nest_model.cuda()

	return nest_model       
        

        
def main():

    global opt, model, netContent
    opt = parser.parse_args()
    print(opt)
    record_parser(opt)

    cuda = opt.cuda
    if cuda:
        print("=> use gpu id: '{}'".format(opt.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
        if not torch.cuda.is_available():
                raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True

    print("===> Loading datasets")

    
    FDG_root=opt.dtstCh1
    FDG_dataset_list=os.listdir(FDG_root)
    for n in range(len(FDG_dataset_list)):
        FDG_dataset_list[n]=os.path.join(FDG_root,FDG_dataset_list[n])     

    MRI_root=opt.dtstCh2
    MRI_dataset_list=os.listdir(MRI_root)
    for n in range(len(MRI_dataset_list)):
        MRI_dataset_list[n]=os.path.join(MRI_root,MRI_dataset_list[n])     

    dataset = Data__Set(root0=FDG_dataset_list,root1=MRI_dataset_list,root2=MRI_dataset_list,transform=trns.Compose([trns.ToTensor()]))
    testloader = torch.utils.data.DataLoader(dataset,batch_size=opt.batchSize)
   


    print("===> Building model")
    input_nc = 1
    output_nc = 1
    #nb_filter = [64, 112, 160, 208, 256]
    nb_filter = [64, 96, 128, 256]
    deepsupervision = False


    net = Net(nb_filter, input_nc, output_nc, deepsupervision, fusion_type=opt.fusion, scale_head=opt.heads)

    if opt.resume:
        if os.path.isfile(opt.resume):
            net.load_state_dict(torch.load(opt.resume))
            print(torch.load(opt.resume).keys())
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))
            sys.exit("sorry, goodbye!")

            

    para = sum([np.prod(list(p.size())) for p in net.parameters()])
    type_size = 4
    print('Model {} : params: {:4f}M'.format(net._get_name(), para * type_size / 1000 / 1000))



    inference(dataset,testloader, net, opt.start_epoch)



    
    
    
def inference(dataset,dataloader, model, epoch):

    
    model.eval()

    
    fusion_type = ['attention_avg', 'attention_max', 'attention_nuclear']

    for iteration, data in enumerate(dataloader, 1):
        
        batch=data[0]
        batch_path=data[1]
        
        FDG_1c = Variable(batch[:,0,0,:,:],requires_grad=True)
        FDG_1c = FDG_1c[:,None,:,:]
        if opt.maskMRI:
            MRI_1c = Variable(batch[:,1,1,:,:], requires_grad=True)
        else:
            MRI_1c = Variable(batch[:,1,0,:,:], requires_grad=True)
        MRI_1c = MRI_1c[:,None,:,:]
        

        if opt.cuda:
            model.cuda()
            FDG_1c = FDG_1c.cuda()
            MRI_1c = MRI_1c.cuda()

    #-----------model
        # encoder
        if opt.encoder:
            en_FDG = model.encoder1(FDG_1c)
            en_MRI = model.encoder2(MRI_1c)
        else:
            en_FDG = model.encoder(FDG_1c)
            en_MRI = model.encoder(MRI_1c)
        # fusion
        
        f_type=fusion_type[0]
        #f = model.fusion(en_FDG, en_MRI, f_type)
        
        
        f = model.fusion(en_FDG, en_MRI)
            
        # decoder
        outputs = model.decoder_train(f)
        #print(outputs.type)
        

    #-----------model

        MRI_1c = Variable(batch[:,1,0,:,:], requires_grad=True)
        MRI_1c = MRI_1c[:,None,:,:]
        FDG_1c_cpu= np.asarray(FDG_1c.cpu().detach())
        outputs_cpu= np.asarray(outputs.cpu().detach())
        MRI_1c_cpu= np.asarray(MRI_1c.cpu().detach())
        
        for n in range(len(outputs_cpu)):
            X=iteration*16+n-16
            print(X)
            #print(dataset[X])
            fused=Image.fromarray(outputs_cpu[n,0,:,:])
            source1=Image.fromarray(MRI_1c_cpu[n,0,:,:])
            source2=Image.fromarray(FDG_1c_cpu[n,0,:,:])
            
            print(batch_path[n])
            fused_filename=batch_path[n][35:-4]+".tif"#str(X)+"_fused.tif"
            source1_filename=batch_path[n][35:-4]+".tif"
            source2_filename=batch_path[n][35:-4]+".tif"
            #print(fused_filename,source1_filename,source2_filename)
            #f=os.path.split(opt.resume)
            f=opt.resume.split('/')
            print(f)
            print(f[0],f[1],f[2])
            filefolder=f[1]+"_"+f[2][12:]
            filefolder=f[0]+"_"+f[2]
            
            #output
            if not os.path.exists(os.path.join(output_folder,filefolder)):
                os.makedirs(os.path.join(output_folder,filefolder))
            if not os.path.exists(os.path.join(output_folder,filefolder,"fused")):
                os.makedirs(os.path.join(output_folder,filefolder,"fused"))
            if not os.path.exists(os.path.join(output_folder,filefolder,"source1")):
                os.makedirs(os.path.join(output_folder,filefolder,"source1"))
            if not os.path.exists(os.path.join(output_folder,filefolder,"source2")):
                os.makedirs(os.path.join(output_folder,filefolder,"source2"))
                
            fused.save(os.path.join(output_folder,filefolder,"fused", fused_filename))
            source1.save(os.path.join(output_folder,filefolder,"source1", source1_filename))
            source2.save(os.path.join(output_folder,filefolder,"source2", source2_filename))
            
            #vis
            '''if not os.path.exists(os.path.join(output_folder,filefolder)):
                os.makedirs(os.path.join(output_folder,filefolder))
            fused.save(os.path.join(output_folder,filefolder,str(X)+"_fused.tif")) 
            source1.save(os.path.join(output_folder,filefolder,str(X)+"_source1.tif"))
            source2.save(os.path.join(output_folder,filefolder,str(X)+"_source2.tif"))'''
            
            
            '''print(batch_path[n])
            plt.figure()
            plt.subplot(3,1,1)
            plt.imshow(outputs_cpu[n,0,:,:],cmap='gray')
            plt.subplot(3,1,2)
            plt.imshow(FDG_1c_cpu[n,0,:,:],cmap='gray')
            plt.subplot(3,1,3)
            plt.imshow(MRI_1c_cpu[n,0,:,:],cmap='gray')
            plt.show()'''


if __name__ == "__main__":
    main()            
