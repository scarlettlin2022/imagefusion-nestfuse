import argparse, os
import torch
import math, random
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from data__set import Data__Set, PairedDataSet, TriDataSet
import time   
import matplotlib.pyplot as plt
import torch.utils.data as TD
import pandas as pd
import numpy as np
import math
from torch.optim import Adam
from sklearn.model_selection import KFold
import torchvision.transforms as trns

from args_fusion import args
import pytorch_msssim
from module import SaveBestModel, GradientLoss, AdaptiveMSE, AdaptiveSSIM
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM

from net_UNetPP import NestFuse
from net_SwinUNet import SwinUNet
from net_SwinFusion import SwinFusion 
from net_SwinFuse import SwinFuse

import torch.nn as nn

from tqdm import tqdm

# Training settings
parser = argparse.ArgumentParser(description="PyTorch SRResNet")
parser.add_argument("--cuda", action="store_true", help="Use cuda?")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
parser.add_argument("--threads", type=int, default=0, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--step", type=int, default=10, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")#mask_npy_111
'''parser.add_argument("--dtstCh1", default='src/FDG_112_train', type=str, help="root folder: channel 1 of input dataset")
parser.add_argument("--dtstCh2", default='src/MRI_112_train', type=str, help="root folder: channel 2 of input dataset")'''


parser.add_argument("--batchSize", type=int, default=16, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=500, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")


parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")


parser.add_argument("--encoder", action="store_true",default=False ,  help="joint(0, Default)/seperate(1) encoder network")
parser.add_argument("--maskMRI", action="store_true",default=False ,  help="full MRI(0, Default)/grey matter only(1)")
parser.add_argument("--invertedMRI", action="store_true",default=False ,  help="full MRI(0, Default)/inverted(1)")
parser.add_argument("--gradient", action="store_true",default=False ,  help="without gradient loss(0, Default)/with gradient loss(1)")
parser.add_argument("--adaptive", action="store_true",default=False ,  help="without adaptive ssim and picxel loss(0, Default)/with adaptive ssim and picxel loss(1)")
parser.add_argument("--fusion", type=int, default=0,  help="spatial_channel(0, Default)/efficient(1)/linear(2)/hydra(3)")
parser.add_argument("--heads", type=int, default=4, help="number of attention heads (recommend 1~16)")

parser.add_argument("--network", type=int, default=0,  help="UNet++(0, Default)/SwinUNet(1)/SwinFusion(2)/SwinFuse(3)")
parser.add_argument("--fold", nargs='+', help='select lists of folds', required=True)
parser.add_argument("--folder", type=str, default="unknown", help="please enter the type of model as folder name, expect the results of five fold all stored in one folder")

parser.add_argument("--ssim2pixel", type=float, default=10,  help="ssim to pixel loss ratio")
parser.add_argument("--PET2MRIssim", type=float, default=5,  help="PET to MRI ssim loss ratio")
parser.add_argument("--PET2MRIpixel", type=float, default=10,  help="PET to MRI pixel loss ratio")

parser.add_argument("--scalingMRI", type=float, default=1,  help="scaling of MRI content")
parser.add_argument("--scalingPET", type=float, default=1,  help="scaling of PET content")

parser.add_argument("--slice_norm", action="store_true",default=False ,  help="MRI norm type: 3d norm(0, Default)/slice norm(1)")

global style_weights,content_weight,style_weight,layer_dict



def psnr(target, ref, scale=None):
    target_data = np.array(target)
    ref_data = np.array(ref)
    diff = ref_data - target_data
    diff = diff.flatten()
    rmse = math.sqrt( np.mean(diff ** 2.) )
    return 20*math.log10(1.0/rmse)


def record_parser(opt):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    f = open(os.path.join(output_folder,"command.txt"), 'w')
    f.write(str(opt))
    f.close()
    
        

                
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10"""
    lr = opt.lr * (0.98 ** (epoch // opt.step))#0.1
    return lr 

def record_loss_curve(x,y1,y2,title):
    plt.figure()
    plt.plot(x,y1,label='train loss')
    plt.plot(x,y2,label='test loss')
    plt.title(title)
    plt.legend(loc='lower left')
    filename=title+".png"
    filefolder=output_folder+"loss_curve/"

    if not os.path.exists(filefolder):
        os.makedirs(filefolder)
    plt.savefig(os.path.join(filefolder,filename))
    plt.close()    
        
def save_checkpoint(model, epoch, weight, fold):
    model_out_path = output_folder+"checkpoint/" + "model_fold_{}_epoch_{}_{}.pth".format(fold,epoch, weight)
    state = {"epoch": epoch ,"model": model}
    if not os.path.exists(output_folder+"checkpoint/"):
        os.makedirs(output_folder+"checkpoint/")

    torch.save(model.state_dict(), model_out_path)#state

    print("Checkpoint saved to {}".format(model_out_path))
        
def main():

    global opt, model, netContent, output_folder

    opt = parser.parse_args()
    
    

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

#DATA LOADING

    print("===> Loading datasets")

    for fold in opt.fold:
        
        output_folder="model/"+opt.folder+"/fold"+str(fold)+"/"+str(time.localtime()[0])+"-"+str(time.localtime()[1])+"-"+str(time.localtime()[2])+"-"+str(time.localtime()[3])+str(time.localtime()[4])+str(time.localtime()[5])+'/'
        if not os.path.exists( output_folder):
            os.makedirs( output_folder)
        print('output folder', output_folder)
        
        print(opt)
        record_parser(opt)
        
        print(print("==================FOLD {}==================".format(fold)))
        
        #-----------------------------------------
        #train
        MRI_train=[]
        FDG_train=[] 
        MRI_inverted_train=[]

        f_name='src/fold/fold0/train_MRI.txt'
        if opt.slice_norm:
            f_name.replace('fold/', 'fold_112_slice_norm/')
            print("-------------MRI input using slice norm")
        f = open(f_name, 'r')  
        for line in f.readlines():
            MRI_train.append(str(line)[:-2])
            #print(str(line)[:-2])
            
        f_name='src/fold/fold0/train_PET.txt'
        if opt.slice_norm:
            f_name.replace('fold/', 'fold_112_slice_norm/')
            print("-------------MRI input using slice norm")
        f = open(f_name, 'r')
        for line in f.readlines():
            FDG_train.append(str(line)[:-2])
            
        f_name='src/fold/fold0/train_MRI.txt'
        f = open(f_name, 'r')  
        for line in f.readlines():
            MRI_inverted_train.append(str(line)[:-2].replace('MRI_112/', 'MRI_112_inverted/'))
        #-----------------------------------------
        #valid
        MRI_valid=[]
        FDG_valid=[] 
        MRI_inverted_valid=[]

        f_name='src/fold/fold0/valid_MRI.txt'
        if opt.slice_norm:
            f_name.replace('fold/', 'fold_112_slice_norm/')
            print("-------------MRI input using slice norm")
        f = open(f_name, 'r')  
        for line in f.readlines():
            MRI_valid.append(str(line)[:-2])
            #print(str(line)[:-2])
            
        f_name='src/fold/fold0/valid_PET.txt'
        if opt.slice_norm:
            f_name.replace('fold/', 'fold_112_slice_norm/')
            print("-------------MRI input using slice norm")
        f = open(f_name, 'r')
        for line in f.readlines():
            FDG_valid.append(str(line)[:-2])
            
        f_name='src/fold/fold0/valid_MRI.txt'
        f = open(f_name, 'r')  
        for line in f.readlines():
            MRI_inverted_valid.append(str(line)[:-2].replace('MRI_112/', 'MRI_112_inverted/'))
            #print(str(line)[:-2].replace('MRI_112/', 'MRI_112_inverted/'))
        
        '''#-----------------------------------------
        #test
        MRI_test=[]
        FDG_test=[] 
        MRI_inverted_test=[]

        f_name='src/fold/fold0/test_MRI.txt'
        if opt.slice_norm:
            f_name.replace('fold/', 'fold_112_slice_norm/')
            print("-------------MRI input using slice norm")
        f = open(f_name, 'r')  
        for line in f.readlines():
            MRI_test.append(str(line)[:-2])
            print(str(line)[:-2])
            
        f_name='src/fold/fold0/test_PET.txt'
        if opt.slice_norm:
            f_name.replace('fold/', 'fold_112_slice_norm/')
            print("-------------MRI input using slice norm")
        f = open(f_name, 'r')
        for line in f.readlines():
            FDG_test.append(str(line)[:-2])
            
        f_name='src/fold/fold0/test_MRI.txt'
        f = open(f_name, 'r')  
        for line in f.readlines():
                MRI_inverted_test.append(str(line)[:-2].replace('MRI_112/', 'MRI_112_inverted/'))'''
        
        
        train_dataset = TriDataSet(root0=FDG_train, root1=MRI_train, root2=MRI_inverted_train, transform=trns.Compose([trns.ToTensor()]))
        valid_dataset = TriDataSet(root0=FDG_valid, root1=MRI_valid, root2=MRI_inverted_valid, transform=trns.Compose([trns.ToTensor()]))
        
        trainloader = torch.utils.data.DataLoader(
                        train_dataset, 
                        batch_size=opt.batchSize, shuffle=True)
        validloader = torch.utils.data.DataLoader(
                        valid_dataset,
                        batch_size=opt.batchSize, shuffle=True)
        
        '''MRI_train=[]
        FDG_train=[] 
        MRI_valid=[]
        FDG_valid=[] 
        
        f_name='src/fold/fold{}/train_MRI.txt'.format(str(fold))
        if opt.slice_norm:
            f_name.replace('fold/', 'fold_112_slice_norm/')
            print("-------------MRI input using slice norm")
        f = open(f_name, 'r')   
        for line in f.readlines():
            MRI_train.append(str(line)[:-2])
            #print(str(line[:-1]))
            
        f_name='src/fold/fold{}/train_PET.txt'.format(str(fold))
        if opt.slice_norm:
            f_name.replace('fold/', 'fold_112_slice_norm/')
        f = open(f_name, 'r')   
        for line in f.readlines():
            FDG_train.append(str(line)[:-2])
            
        f_name='src/fold/fold{}/valid_MRI.txt'.format(str(fold))
        if opt.slice_norm:
            f_name.replace('fold/', 'fold_112_slice_norm/')
        f = open(f_name, 'r')  
        for line in f.readlines():
            MRI_valid.append(str(line)[:-2])
            #print(str(line[:-1]))
            
        f_name='src/fold/fold{}/valid_PET.txt'.format(str(fold))
        if opt.slice_norm:
            f_name.replace('fold/', 'fold_112_slice_norm/')
        f = open(f_name, 'r')
        for line in f.readlines():
            FDG_valid.append(str(line)[:-2])
        
        train_dataset = PairedDataSet(root0=FDG_train,root1=MRI_train,transform=trns.Compose([trns.ToTensor()]))
        valid_dataset = PairedDataSet(root0=FDG_valid,root1=MRI_valid,transform=trns.Compose([trns.ToTensor()]))
        
        # Define data loaders for training and testing data in this fold
        trainloader = torch.utils.data.DataLoader(
                        train_dataset, 
                        batch_size=opt.batchSize, shuffle=True)
        validloader = torch.utils.data.DataLoader(
                        valid_dataset,
                        batch_size=opt.batchSize, shuffle=True)'''

#MODEL

        print("===> Building model")

        if opt.network==0:
            input_nc = 1
            output_nc = 1
            #nb_filter = [64, 112, 160, 208, 256]
            nb_filter = [64, 96, 128, 256]
            deepsupervision = False
            net = NestFuse(nb_filter, input_nc, output_nc, deepsupervision, fusion_type=opt.fusion, scale_head=opt.heads)
            
            para = sum([np.prod(list(p.size())) for p in net.parameters()])
            type_size = 4
            print('Model {} : params: {:4f}M'.format(net._get_name(), para * type_size / 1000 / 1000))
            
        
            print("-------------currently using unet++ as backbone----------------------")
        elif opt.network==1:

            img_size=112
            patch_size=2#1
            in_chans=1
            out_chans=in_chans
            num_classes=1
            embed_dim=96
            depths=[2, 2, 2, 2]
            depths_decoder=[1, 2, 2, 2]
            num_heads=[3, 6, 12, 24]
            window_size=7
            mlp_ratio=4.
            qkv_bias=True
            qk_scale=None
            drop_rate=0.
            attn_drop_rate=0.
            drop_path_rate=0.1
            norm_layer=nn.LayerNorm
            ape=True
            patch_norm=True
            use_checkpoint=False    
            
            
            net=SwinUNet(img_size=img_size,
                                patch_size=patch_size,
                                in_chans=in_chans,
                                num_classes=out_chans,
                                embed_dim=embed_dim,
                                depths=depths,
                                num_heads=num_heads,
                                window_size=window_size,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias,
                                qk_scale=qk_scale,
                                drop_rate=drop_rate,
                                drop_path_rate=drop_path_rate,
                                ape=ape,
                                patch_norm=patch_norm,
                                use_checkpoint=use_checkpoint)
            print("-------------currently using swinunet as backbone----------------------")
        elif opt.network==2:
            patch_size=1#2 
            upscale=1
            in_chans=1
            img_size=112 
            window_size=7#8 #7
            img_range=1.
            
            
            embed_dim=60 #96
            num_heads=[6, 6, 6, 6]
            mlp_ratio=2#4., 
            upsampler="null"
            resi_connection='1conv'
            
            net = SwinFusion(upscale=upscale, img_size= img_size, in_chans=in_chans, patch_size=patch_size,window_size=window_size, img_range=img_range,embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,upsampler=upsampler,resi_connection=resi_connection)
            print("-------------currently using SwinFusion as backbone----------------------")
        
        elif opt.network==3:
            img_size = 112
            patch_size = 1
            
            net = SwinFuse(img_size=img_size, patch_size=patch_size)
            print("-------------currently using SwinFuse as backbone----------------------")
            

        if opt.resume:
            if os.path.isfile(opt.resume):
                print("=> loading checkpoint '{}'".format(opt.resume))
                net.load_state_dict(torch.load(opt.resume))
            else:
                print("=> no checkpoint found at '{}'".format(opt.resume))

        optimizer = Adam(net.parameters(), opt.lr)
        
#CHECKPOINT
        
        train_df = pd.DataFrame(np.zeros((opt.nEpochs, 7)),columns =['total', 'FDG_pixel', 'FDG_ssim', 'FDG_gradient', 'MRI_pixel', 'MRI_ssim', 'MRI_gradient'])
        valid_df = pd.DataFrame(np.zeros((opt.nEpochs, 7)),columns =['total', 'FDG_pixel', 'FDG_ssim', 'FDG_gradient', 'MRI_pixel', 'MRI_ssim', 'MRI_gradient'])
        save_best_loss_model = SaveBestModel(name='loss', minmax=0, output_folder=output_folder)
        print("===> Training")
        for epoch in range(opt.start_epoch, opt.nEpochs + 1):
            '''for i in range(2,3):
                ratio_ssim=args.ssim_weight[i]'''

#TRAINING, VALIDATION
                
            train(trainloader, optimizer, net, epoch, fold, train_df)
            if epoch%20==0:
                save_checkpoint(net, epoch, opt.ssim2pixel, fold)
                x=range(1,epoch+1)
                record_loss_curve(x,np.asarray(train_df.iloc[0:epoch]["total"]),np.asarray(valid_df.iloc[0:epoch]["total"]), "total_loss_fold_"+str(fold)+"_epoch_"+str(epoch))
            valid(validloader, save_best_loss_model, net, epoch, fold, valid_df) 


def train(trainloader, optimizer, model, epoch, fold, train_df):
        
    lr = adjust_learning_rate(optimizer, epoch-1)
    
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    print("Epoch={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))

    model.train()

    all_pixel_loss_FDG = 0.
    all_pixel_loss_MRI = 0.
    all_ssim_loss_FDG = 0.
    all_ssim_loss_MRI = 0.
    all_total_loss = 0.
    all_gd_loss_FDG = 0.
    all_gd_loss_MRI = 0.
    if opt.gradient:
        gd_loss = GradientLoss()
    if opt.adaptive:
        mse_loss = AdaptiveMSE()
        ssim_loss = AdaptiveSSIM()
    else:
        mse_loss = torch.nn.MSELoss()

    
    fusion_type = ['attention_avg', 'attention_max', 'attention_nuclear']

    for iteration, data in enumerate(trainloader, 1):
        
        batch=data[0]
        batch_path=data[1]
        #print(batch_path)
        #print(batch.shape)
        
        FDG_1c = Variable(batch[:,0,0,:,:],requires_grad=True)
        FDG_1c = FDG_1c[:,None,:,:]
        
        
        if opt.maskMRI==True and opt.invertedMRI==False:
            MRI_gm = Variable(batch[:,1,1,:,:], requires_grad=True)
        if opt.maskMRI==False and opt.invertedMRI==False:
            MRI_gm = Variable(batch[:,1,0,:,:], requires_grad=True)
        if opt.maskMRI==True and opt.invertedMRI==True:
            MRI_gm = Variable(batch[:,2,1,:,:], requires_grad=True)
        if opt.maskMRI==False and opt.invertedMRI==True:
            MRI_gm = Variable(batch[:,2,0,:,:], requires_grad=True)
            
            
        MRI_gm = MRI_gm[:,None,:,:]
        
        MRI_1c = Variable(batch[:,1,0,:,:], requires_grad=True)
        
        MRI_1c = MRI_1c[:,None,:,:]
        
        MRI_1c_scaled = MRI_1c*opt.scalingMRI
        FDG_1c_scaled = FDG_1c*opt.scalingPET
        MRI_gm_scaled = MRI_gm*opt.scalingMRI

        
        if opt.cuda:
            model.cuda()
            FDG_1c = FDG_1c.cuda()
            MRI_1c = MRI_1c.cuda()
            MRI_gm = MRI_gm.cuda()
            MRI_1c_scaled = MRI_1c_scaled.cuda()
            FDG_1c_scaled = FDG_1c_scaled.cuda()
            MRI_gm_scaled = MRI_gm_scaled.cuda()
        #print("input", torch.mean(FDG_1c),torch.mean(MRI_1c))
        #if iteration %100==0:#if batch_path[0].find('941')!=-1 or batch_path[1].find('941')!=-1:#i
        
            
            
        if opt.network==0:
            # encoder
            if opt.encoder:
                en_FDG = model.encoder1(FDG_1c_scaled)
                en_MRI = model.encoder2(MRI_gm_scaled)
            else:
                en_FDG = model.encoder(FDG_1c_scaled)
                en_MRI = model.encoder(MRI_gm_scaled)
            # fusion
            f_type=fusion_type[0]
            f = model.fusion(en_FDG, en_MRI)
            # decoder
            outputs = model.decoder_train(f)
        elif opt.network==1:
            x, x_downsample = model.forward_features(FDG_1c_scaled)
            y, y_downsample = model.forward_features(MRI_gm_scaled)
            #print(x.shape, x_downsample.shape)
            xy_downsample = model.feature_fusion(x_downsample, y_downsample)
            xy = model.forward_up_features(xy_downsample[-1],xy_downsample)
            outputs = model.up_x4(xy)
        elif opt.network==2 or opt.network==3:
            #x = torch.randn((opt.batchSize, 1, 112, 112)).cuda()
            #y = torch.randn((opt.batchSize, 1, 112, 112)).cuda()
            outputs = model(FDG_1c_scaled,MRI_gm_scaled)#(FDG_1c,MRI_gm)(x,y)
            #print(torch.mean(outputs))
            #print(torch.mean(FDG_1c), torch.mean(MRI_gm))
        #outputs=model(FDG_1c,MRI_gm)
        
        '''x, x_downsample = model.forward_features(FDG_1c)
        y, y_downsample = model.forward_features(MRI_gm)
        xy_downsample = model.feature_fusion(x_downsample, y_downsample)
        xy = model.forward_up_features(xy_downsample[-1],xy_downsample)
        outputs = model.up_x4(xy)'''
        
        #info_ratio(FDG_1c,MRI_1c, outputs)
        if iteration+1 ==len(trainloader):
            FDG_1c_cpu=FDG_1c.cpu().detach()
            MRI_1c_cpu=MRI_1c.cpu().detach()
            outputs_cpu=outputs.cpu().detach()
            plt.figure(figsize=(80,60))  
            for i in range(4):
                plt.subplot(4,3,3*i+1)
                plt.axis('off')
                plt.imshow(FDG_1c_cpu[i,0])
                plt.subplot(4,3,3*i+2)
                plt.axis('off')
                plt.imshow(MRI_1c_cpu[i,0],cmap='gray')
                plt.subplot(4,3,3*i+3)
                plt.axis('off')
                plt.imshow(outputs_cpu[i,0])
            filename="train_epoch"+str(epoch)+".png"
            filefolder=output_folder+"output_map/"
        
            if not os.path.exists(filefolder):
                os.makedirs(filefolder)
            plt.savefig(os.path.join(filefolder,filename))
            plt.close()
        ssim_loss_value = 0.
        pixel_loss_value = 0.
        gradient_loss_value = 0.
        
        '''pixel_loss_FDG = 0
        pixel_loss_MRI = 0
        ssim_loss_FDG = 0
        ssim_loss_MRI = 0'''
        
        if opt.adaptive:
            pixel_loss_FDG, pixel_loss_MRI = mse_loss(outputs, FDG_1c_scaled, MRI_1c_scaled)
            ssim_loss_FDG, ssim_loss_MRI = ssim_loss(outputs, FDG_1c_scaled, MRI_1c_scaled)
        else:
            pixel_loss_FDG = mse_loss(outputs, FDG_1c_scaled)
            pixel_loss_MRI = mse_loss(outputs, MRI_1c_scaled)

            ssim_loss_FDG = (1-pytorch_msssim.msssim(outputs, FDG_1c_scaled))
            ssim_loss_MRI = (1-pytorch_msssim.msssim(outputs, MRI_1c_scaled))
            
            '''pixel = mse_loss(MRI_1c, FDG_1c)
            sim = (1-pytorch_msssim.msssim(MRI_1c, FDG_1c))'''
            #print(batch_path[0])
            '''ssim=SSIM(data_range=1.0).cuda()
            ssim_FDG=ssim(outputs, FDG_1c)
            ssim_MRI=ssim(outputs, MRI_1c)
            sim=ssim(FDG_1c, MRI_1c)'''
            
            #print(pixel_loss_FDG, pixel_loss_MRI, ssim_loss_FDG, ssim_loss_MRI, pixel, sim)
            
        if opt.gradient:
            
            gd_loss_FDG=gd_loss(outputs, FDG_1c_scaled)
            gd_loss_MRI=gd_loss(outputs, MRI_1c_scaled)
        else:
            gd_loss_FDG=0
            gd_loss_MRI=0
        
        
        pixel_loss_value += pixel_loss_FDG*opt.PET2MRIpixel
        pixel_loss_value += pixel_loss_MRI

        ssim_loss_value += ssim_loss_FDG*opt.PET2MRIssim
        ssim_loss_value += ssim_loss_MRI
        
        gradient_loss_value += 0.2*gd_loss_FDG
        gradient_loss_value += 0.8*gd_loss_MRI
        
            
        # total loss
        '''if opt.adaptive:
            #---->only FDG SSIM decreasing
            #total_loss = 20*pixel_loss_value + ssim_loss_value + gradient_loss_value
            
            #total_loss = 20*pixel_loss_value + 10*ssim_loss_value + gradient_loss_value
            
            total_loss = pixel_loss_value + 10*ssim_loss_value + gradient_loss_value
        else:'''

        total_loss = pixel_loss_value + opt.ssim2pixel * ssim_loss_value + gradient_loss_value

            
    
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        all_pixel_loss_FDG += pixel_loss_FDG.item()
        all_pixel_loss_MRI += pixel_loss_MRI.item()
        all_ssim_loss_FDG += ssim_loss_FDG.item()
        all_ssim_loss_MRI += ssim_loss_MRI.item()
        if opt.gradient:
            all_gd_loss_FDG += gd_loss_FDG.item()
            all_gd_loss_MRI += gd_loss_MRI.item()
        all_total_loss += total_loss.item()
        
    

        if (iteration + 1) % args.log_interval == 0:

            print("train-> Fold {}, Epoch {} [{}/{}]:\t pixel loss FDG: {:.6f}, pixel loss MRI: {:.6f}, ssim loss FDG: {:.6f}, ssim loss MRI: {:.6f}, gradient loss FDG: {:.6f}, gradient loss MRI: {:.6f}, total: {:.6f}\n".format(fold,epoch, iteration, len(trainloader), pixel_loss_FDG, pixel_loss_MRI, ssim_loss_FDG, ssim_loss_MRI, gd_loss_FDG, gd_loss_MRI, total_loss.item()))
            
            train_df.iloc[epoch-1]=[all_total_loss/(iteration + 1), all_pixel_loss_FDG/(iteration + 1), all_ssim_loss_FDG/(iteration + 1), all_gd_loss_FDG/(iteration + 1), all_pixel_loss_MRI/(iteration + 1), all_ssim_loss_MRI/(iteration + 1), all_gd_loss_MRI/(iteration + 1)]
            
            '''xy_downsample_cpu=xy_downsample[0].cpu().detach()
            plt.figure(figsize=(80,60))  
            for i in range(4):
                plt.subplot(2,2,i+1)
                plt.axis('off')
                plt.imshow(xy_downsample_cpu[i])
            filename="epoch"+str(epoch)+"_"+str(iteration)+".png"
            filefolder=output_folder+"output_map/"
            
            if not os.path.exists(filefolder):
                os.makedirs(filefolder)
            plt.savefig(os.path.join(filefolder,filename))
            plt.close()'''
            
            '''filename="epoch"+str(epoch)+"_"+str(iteration)+".png"
            filefolder=output_folder+"output_map/"
            
            if not os.path.exists(filefolder):
                os.makedirs(filefolder)
            plt.savefig(os.path.join(filefolder,filename))
            plt.close()'''
            
        #break
    excel_file=output_folder+'train_'+str(fold)+'.xlsx'
    train_df.to_excel(excel_file)


    all_pixel_loss_FDG = 0.
    all_pixel_loss_MRI = 0.
    all_ssim_loss_FDG = 0.
    all_ssim_loss_MRI = 0. 
    all_gd_loss_FDG = 0.
    all_gd_loss_MRI = 0. 
    all_total_loss = 0.
        

          
def valid(validloader, save_best_loss_model, model, epoch, fold, valid_df):
        
    model.eval()

    all_pixel_loss_FDG = 0.
    all_pixel_loss_MRI = 0.
    all_ssim_loss_FDG = 0.
    all_ssim_loss_MRI = 0.
    all_total_loss = 0.
    all_gd_loss_FDG = 0.
    all_gd_loss_MRI = 0.
    if opt.gradient:
        gd_loss = GradientLoss()
    if opt.adaptive:
        mse_loss = AdaptiveMSE()
        ssim_loss = AdaptiveSSIM()
    else:
        mse_loss = torch.nn.MSELoss()
    
    fusion_type = ['attention_avg', 'attention_max', 'attention_nuclear']

    for iteration, data in enumerate(validloader, 1):
        
        batch=data[0]
        batch_path=data[1]
        #rint(iteration, batch_path)
        FDG_1c = Variable(batch[:,0,0,:,:],requires_grad=True)
        FDG_1c = FDG_1c[:,None,:,:]
        '''if opt.maskMRI:
            MRI_gm = Variable(batch[:,1,1,:,:], requires_grad=True)
        else:
            MRI_gm = Variable(batch[:,1,0,:,:], requires_grad=True)'''
            
            
        if opt.maskMRI==True and opt.invertedMRI==False:
            MRI_gm = Variable(batch[:,1,1,:,:], requires_grad=True)
        if opt.maskMRI==False and opt.invertedMRI==False:
            MRI_gm = Variable(batch[:,1,0,:,:], requires_grad=True)
        if opt.maskMRI==True and opt.invertedMRI==True:
            MRI_gm = Variable(batch[:,2,1,:,:], requires_grad=True)
        if opt.maskMRI==False and opt.invertedMRI==True:
            MRI_gm = Variable(batch[:,2,0,:,:], requires_grad=True)
            
            
        MRI_gm = MRI_gm[:,None,:,:]
        
        MRI_1c = Variable(batch[:,1,0,:,:], requires_grad=True)
        MRI_1c = MRI_1c[:,None,:,:]
        
        MRI_1c_scaled = MRI_1c*opt.scalingMRI
        FDG_1c_scaled = FDG_1c*opt.scalingPET
        MRI_gm_scaled = MRI_gm*opt.scalingMRI

        
        if opt.cuda:
            model.cuda()
            FDG_1c = FDG_1c.cuda()
            MRI_1c = MRI_1c.cuda()
            MRI_gm = MRI_gm.cuda()
            MRI_1c_scaled = MRI_1c_scaled.cuda()
            FDG_1c_scaled = FDG_1c_scaled.cuda()
            MRI_gm_scaled = MRI_gm_scaled.cuda()
        #print("input", torch.mean(FDG_1c),torch.mean(MRI_1c))
        #if iteration %100==0:#if batch_path[0].find('941')!=-1 or batch_path[1].find('941')!=-1:#i
        
            
            
        if opt.network==0:
            # encoder
            if opt.encoder:
                en_FDG = model.encoder1(FDG_1c_scaled)
                en_MRI = model.encoder2(MRI_gm_scaled)
            else:
                en_FDG = model.encoder(FDG_1c_scaled)
                en_MRI = model.encoder(MRI_gm_scaled)
            # fusion
            f_type=fusion_type[0]
            f = model.fusion(en_FDG, en_MRI)
            # decoder
            outputs = model.decoder_train(f)
        elif opt.network==1:
            x, x_downsample = model.forward_features(FDG_1c_scaled)
            y, y_downsample = model.forward_features(MRI_gm_scaled)
            #print(x.shape, x_downsample.shape)
            xy_downsample = model.feature_fusion(x_downsample, y_downsample)
            xy = model.forward_up_features(xy_downsample[-1],xy_downsample)
            outputs = model.up_x4(xy)
        elif opt.network==2 or opt.network==3:
            outputs = model(FDG_1c_scaled,MRI_gm_scaled)
        
        ssim_loss_value = 0.
        pixel_loss_value = 0.
        gradient_loss_value = 0.

        if opt.adaptive:
            pixel_loss_FDG, pixel_loss_MRI = mse_loss(outputs, FDG_1c_scaled, MRI_1c_scaled)
            ssim_loss_FDG, ssim_loss_MRI = ssim_loss(outputs, FDG_1c_scaled, MRI_1c_scaled)
        else:
            pixel_loss_FDG = mse_loss(outputs, FDG_1c_scaled)
            pixel_loss_MRI = mse_loss(outputs, MRI_1c_scaled)

            ssim_loss_FDG = (1-pytorch_msssim.msssim(outputs, FDG_1c_scaled))
            ssim_loss_MRI = (1-pytorch_msssim.msssim(outputs, MRI_1c_scaled))
            
        if opt.gradient:
            
            gd_loss_FDG=gd_loss(outputs, FDG_1c_scaled)
            gd_loss_MRI=gd_loss(outputs, MRI_1c_scaled)
        else:
            gd_loss_FDG=0
            gd_loss_MRI=0
        
        
        #pixel_loss_value += 0.8*pixel_loss_FDG#*pixelRatio
        #pixel_loss_value += 0.2*pixel_loss_MRI
        
        '''pixel_loss_value += pixel_loss_FDG*10
        pixel_loss_value += pixel_loss_MRI

        ssim_loss_value += ssim_loss_FDG*5
        ssim_loss_value += ssim_loss_MRI'''
        
        pixel_loss_value += pixel_loss_FDG*opt.PET2MRIpixel
        pixel_loss_value += pixel_loss_MRI

        ssim_loss_value += ssim_loss_FDG*opt.PET2MRIssim
        ssim_loss_value += ssim_loss_MRI
        
        gradient_loss_value += 0.2*gd_loss_FDG
        gradient_loss_value += 0.8*gd_loss_MRI
        
        # total loss
        #total_loss = pixel_loss_value + ratio_ssim * ssim_loss_value + gradient_loss_value
        # total loss
        '''if opt.adaptive:
            #---->only FDG SSIM decreasing
            #total_loss = 20*pixel_loss_value + ssim_loss_value + gradient_loss_value
            
            #total_loss = 20*pixel_loss_value + 10*ssim_loss_value + gradient_loss_value
        
            total_loss = pixel_loss_value + 10*ssim_loss_value + gradient_loss_value

        else:'''

        total_loss = pixel_loss_value + opt.ssim2pixel * ssim_loss_value + gradient_loss_value


        all_pixel_loss_FDG += pixel_loss_FDG.item()
        all_pixel_loss_MRI += pixel_loss_MRI.item()
        all_ssim_loss_FDG += ssim_loss_FDG.item()
        all_ssim_loss_MRI += ssim_loss_MRI.item()
        if opt.gradient:
            all_gd_loss_FDG += gd_loss_FDG.item()
            all_gd_loss_MRI += gd_loss_MRI.item()
        all_total_loss += total_loss.item()
        

        if (iteration + 1) % args.log_interval == 0:

            print("valid-> Fold {}, Epoch {} [{}/{}]:\t pixel loss FDG: {:.6f}, pixel loss MRI: {:.6f}, ssim loss FDG: {:.6f}, ssim loss MRI: {:.6f}, gradient loss FDG: {:.6f}, gradient loss MRI: {:.6f}, total: {:.6f}\n".format(fold,epoch, iteration, len(validloader), pixel_loss_FDG, pixel_loss_MRI, ssim_loss_FDG, ssim_loss_MRI, gd_loss_FDG, gd_loss_MRI, total_loss.item()))
            
            valid_df.iloc[epoch-1]=[all_total_loss/(iteration + 1), all_pixel_loss_FDG/(iteration + 1), all_ssim_loss_FDG/(iteration + 1), all_gd_loss_FDG/(iteration + 1), all_pixel_loss_MRI/(iteration + 1), all_ssim_loss_MRI/(iteration + 1), all_gd_loss_MRI/(iteration + 1)]
            
        if iteration+1 ==len(validloader):
            FDG_1c_cpu=FDG_1c.cpu().detach()
            MRI_1c_cpu=MRI_1c.cpu().detach()
            outputs_cpu=outputs.cpu().detach()
            plt.figure(figsize=(80,60))  
            for i in range(4):
                plt.subplot(4,3,3*i+1)
                plt.axis('off')
                plt.imshow(FDG_1c_cpu[i,0])
                plt.subplot(4,3,3*i+2)
                plt.axis('off')
                plt.imshow(MRI_1c_cpu[i,0],cmap='gray')
                plt.subplot(4,3,3*i+3)
                plt.axis('off')
                plt.imshow(outputs_cpu[i,0])
            filename="valid_epoch"+str(epoch)+".png"
            filefolder=output_folder+"output_map/"
                
            
            #break
    excel_file=output_folder+'valid_'+str(fold)+'.xlsx'
    valid_df.to_excel(excel_file)

    all_pixel_loss_FDG = 0.
    all_pixel_loss_MRI = 0.
    all_ssim_loss_FDG = 0.
    all_ssim_loss_MRI = 0. 
    all_gd_loss_FDG = 0.
    all_gd_loss_MRI = 0. 
    all_total_loss = 0.
    
    save_best_loss_model(np.mean(np.array(valid_df.iloc[0:epoch]["total"])),epoch, fold, model) 

if __name__ == "__main__":
    
    
    
    
    main()
    '''outputs = torch.Tensor([[[[1, 0.5, 0], [1, 0.5, 0], [1, 0.5, 0]]],[[[1, 0.5, 0], [1, 0.5, 0], [1, 0.5, 0]]]]).cuda()
    targets = torch.Tensor([[[[1, 0.5, 0], [1, 0.5, 0], [1, 0.5, 1]]],[[[1, 0.5, 0], [1, 0.5, 0], [1, 0.5, 1]]]]).cuda()
    print(outputs.shape)
    gd_loss=GradientLoss()
    loss=gd_loss(outputs, targets)
    print(loss)
    
    outputs = torch.Tensor([[[[1, 0.5, 0], [1, 0.5, 0], [1, 0.5, 0]]]]).cuda()
    targets = torch.Tensor([[[[1, 0.5, 0], [1, 0.5, 0], [1, 0.5, 1]]]]).cuda()
    print(outputs.shape)
    gd_loss=GradientLoss()
    loss=gd_loss(outputs, targets)
    print(loss)'''
    
