import argparse, os
import torch
import math, random
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from data__set import Data__Set
import time   
import matplotlib.pyplot as plt
import torch.utils.data as TD
import pandas as pd
import numpy as np
import math
from torch.optim import Adam
from sklearn.model_selection import KFold
import torchvision.transforms as trns


from net_v1 import Net
from args_fusion import args
import pytorch_msssim
from module import SaveBestModel, GradientLoss, AdaptiveMSE, AdaptiveSSIM
from swin_unet import SwinUNet

# Training settings
parser = argparse.ArgumentParser(description="PyTorch SRResNet")
parser.add_argument("--cuda", action="store_true", help="Use cuda?")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
parser.add_argument("--threads", type=int, default=0, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--step", type=int, default=10, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")#mask_npy_111
parser.add_argument("--dtstCh1", default='src/FDG_112_train', type=str, help="root folder: channel 1 of input dataset")
parser.add_argument("--dtstCh2", default='src/MRI_112_train', type=str, help="root folder: channel 2 of input dataset")


parser.add_argument("--batchSize", type=int, default=16, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=500, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")


parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")


parser.add_argument("--encoder", action="store_true",default=False ,  help="joint(0, Default)/seperate(1) encoder network")
parser.add_argument("--maskMRI", action="store_true",default=False ,  help="full MRI(0, Default)/grey matter only(1)")
parser.add_argument("--gradient", action="store_true",default=False ,  help="without gradient loss(0, Default)/with gradient loss(1)")
parser.add_argument("--adaptive", action="store_true",default=False ,  help="without adaptive ssim and picxel loss(0, Default)/with adaptive ssim and picxel loss(1)")
parser.add_argument("--fusion", type=int, default=0,  help="spatial_channel(0, Default)/efficient(1)/linear(2)/hydra(3)")
parser.add_argument("--heads", type=int, default=4, help="number of attention heads (recommend 1~16)")



global style_weights,content_weight,style_weight,layer_dict, output_folder
output_folder=str(time.localtime()[0])+"-"+str(time.localtime()[1])+"-"+str(time.localtime()[2])+"-"+str(time.localtime()[3])+str(time.localtime()[4])+str(time.localtime()[5])+'/'
print('output folder', output_folder)


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
    kfold = KFold(n_splits=5, shuffle=True)
    for fold, (train_ids, valid_ids) in enumerate(kfold.split(dataset)):

        print(train_ids, valid_ids)
        train_subsampler = TD.SubsetRandomSampler(train_ids)
        valid_subsampler = TD.SubsetRandomSampler(valid_ids)
        
        # Define data loaders for training and testing data in this fold
        trainloader = torch.utils.data.DataLoader(
                        dataset, 
                        batch_size=opt.batchSize, sampler=train_subsampler)
        validloader = torch.utils.data.DataLoader(
                        dataset,
                        batch_size=opt.batchSize, sampler=valid_subsampler)

        print("===> Building model")
        input_nc = 1
        output_nc = 1
        #nb_filter = [64, 112, 160, 208, 256]
        nb_filter = [64, 96, 128, 256]
        deepsupervision = False


        net = Net(nb_filter, input_nc, output_nc, deepsupervision, fusion_type=opt.fusion, scale_head=opt.heads)
        
        '''img_size=112
        patch_size=2
        in_chans=1
        out_chans=in_chans
        num_classes=1
        embed_dim=96
        depths=[2, 2, 2, 2]
        depths_decoder=[1, 2, 2, 2]
        num_heads=[3, 6, 12, 24]
        window_size=7
        mlp_ratio=4.
        
        net = SwinUNet(img_size=img_size,
                        patch_size=patch_size,
                        in_chans=in_chans,
                        num_classes=out_chans,
                        embed_dim=embed_dim,
                        depths=depths,
                        num_heads=num_heads,
                        window_size=window_size,
                        mlp_ratio=mlp_ratio)'''

        if opt.resume:
            if os.path.isfile(opt.resume):
                print("=> loading checkpoint '{}'".format(opt.resume))
                net.load_state_dict(torch.load(opt.resume))
            else:
                print("=> no checkpoint found at '{}'".format(opt.resume))


        para = sum([np.prod(list(p.size())) for p in net.parameters()])
        type_size = 4
        print('Model {} : params: {:4f}M'.format(net._get_name(), para * type_size / 1000 / 1000))
        optimizer = Adam(net.parameters(), opt.lr)
        
        train_df = pd.DataFrame(np.zeros((opt.nEpochs, 7)),columns =['total', 'FDG_pixel', 'FDG_ssim', 'FDG_gradient', 'MRI_pixel', 'MRI_ssim', 'MRI_gradient'])
        valid_df = pd.DataFrame(np.zeros((opt.nEpochs, 7)),columns =['total', 'FDG_pixel', 'FDG_ssim', 'FDG_gradient', 'MRI_pixel', 'MRI_ssim', 'MRI_gradient'])
        save_best_loss_model = SaveBestModel(name='loss', minmax=0, output_folder=output_folder)
        print("===> Training")
        for epoch in range(opt.start_epoch, opt.nEpochs + 1):
            for i in range(2,3):
                ratio_ssim=args.ssim_weight[i]
                
                train(ratio_ssim,trainloader, optimizer, net, epoch, fold, train_df)
                valid(ratio_ssim,validloader, save_best_loss_model, net, epoch, fold, valid_df) 

                if epoch%20==0:
                    save_checkpoint(net, epoch, ratio_ssim, fold)
                    x=range(1,epoch+1)
                    record_loss_curve(x,np.asarray(train_df.iloc[0:epoch]["total"]),np.asarray(valid_df.iloc[0:epoch]["total"]), "total_loss_fold_"+str(fold)+"_epoch_"+str(epoch))

def train(ratio_ssim,trainloader, optimizer, model, epoch, fold, train_df):
        
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
        FDG_1c = Variable(batch[:,0,0,:,:],requires_grad=True)
        FDG_1c = FDG_1c[:,None,:,:]
        if opt.maskMRI:
            MRI_gm = Variable(batch[:,1,1,:,:], requires_grad=True)
        else:
            MRI_gm = Variable(batch[:,1,0,:,:], requires_grad=True)
        MRI_gm = MRI_gm[:,None,:,:]
        
        MRI_1c = Variable(batch[:,1,0,:,:], requires_grad=True)
        MRI_1c = MRI_1c[:,None,:,:]
        

        if opt.cuda:
            model.cuda()

            FDG_1c = FDG_1c.cuda()
            MRI_1c = MRI_1c.cuda()
            MRI_gm = MRI_gm.cuda()

        # encoder
        if opt.encoder:
            en_FDG = model.encoder1(FDG_1c)
            en_MRI = model.encoder2(MRI_gm)
        else:
            en_FDG = model.encoder(FDG_1c)
            en_MRI = model.encoder(MRI_gm)
        # fusion
        
        f_type=fusion_type[0]
        f = model.fusion(en_FDG, en_MRI)
        #print(en_FDG[0].shape,en_FDG[1].shape,en_FDG[2].shape,en_FDG[3].shape)
        # decoder
        outputs = model.decoder_train(f)
        
        #outputs=model(FDG_1c,MRI_gm)
        
        '''x, x_downsample = model.forward_features(FDG_1c)
        y, y_downsample = model.forward_features(MRI_gm)
        xy_downsample = model.feature_fusion(x_downsample, y_downsample)
        xy = model.forward_up_features(xy_downsample[-1],xy_downsample)
        outputs = model.up_x4(xy)'''
        
        #info_ratio(FDG_1c,MRI_1c, outputs)

        ssim_loss_value = 0.
        pixel_loss_value = 0.
        gradient_loss_value = 0.
        
        if opt.adaptive:
            pixel_loss_FDG, pixel_loss_MRI = mse_loss(outputs, FDG_1c, MRI_1c)
            ssim_loss_FDG, ssim_loss_MRI = ssim_loss(outputs, FDG_1c, MRI_1c)
        else:
            pixel_loss_FDG = mse_loss(outputs, FDG_1c)*10/ opt.batchSize
            pixel_loss_MRI = mse_loss(outputs, MRI_1c)/ opt.batchSize

            ssim_loss_FDG = (1-pytorch_msssim.msssim(outputs, FDG_1c))/opt.batchSize#/opt.batchSize#, normalize=True
            ssim_loss_MRI = (1-pytorch_msssim.msssim(outputs, MRI_1c))/opt.batchSize#/opt.batchSize#, normalize=True
        if opt.gradient:
            
            gd_loss_FDG=gd_loss(outputs, FDG_1c)
            gd_loss_MRI=gd_loss(outputs, MRI_1c)
        else:
            gd_loss_FDG=0
            gd_loss_MRI=0
        
        
        #pixel_loss_value += 0.8*pixel_loss_FDG#*pixelRatio
        #pixel_loss_value += 0.2*pixel_loss_MRI
        
        pixel_loss_value += pixel_loss_FDG
        pixel_loss_value += pixel_loss_MRI

        ssim_loss_value += ssim_loss_FDG
        ssim_loss_value += ssim_loss_MRI
        
        gradient_loss_value += 0.2*gd_loss_FDG
        gradient_loss_value += 0.8*gd_loss_MRI
        
            
        # total loss
        if opt.adaptive:
            #---->only FDG SSIM decreasing
            #total_loss = 20*pixel_loss_value + ssim_loss_value + gradient_loss_value
            
            #total_loss = 20*pixel_loss_value + 10*ssim_loss_value + gradient_loss_value
            
            total_loss = pixel_loss_value + 10*ssim_loss_value + gradient_loss_value
        else:

            total_loss = pixel_loss_value + ratio_ssim * ssim_loss_value + gradient_loss_value

        
        
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
        

          
def valid(ratio_ssim,validloader, save_best_loss_model, model, epoch, fold, valid_df):
        
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
        
        FDG_1c = Variable(batch[:,0,0,:,:],requires_grad=True)
        FDG_1c = FDG_1c[:,None,:,:]
        if opt.maskMRI:
            MRI_gm = Variable(batch[:,1,1,:,:], requires_grad=True)
        else:
            MRI_gm = Variable(batch[:,1,0,:,:], requires_grad=True)
        MRI_gm = MRI_gm[:,None,:,:]
        
        MRI_1c = Variable(batch[:,1,0,:,:], requires_grad=True)
        MRI_1c = MRI_1c[:,None,:,:]
        

        if opt.cuda:
            model.cuda()

            FDG_1c = FDG_1c.cuda()
            MRI_1c = MRI_1c.cuda()
            MRI_gm = MRI_gm.cuda()

        # encoder
        if opt.encoder:
            en_FDG = model.encoder1(FDG_1c)
            en_MRI = model.encoder2(MRI_gm)
        else:
            en_FDG = model.encoder(FDG_1c)
            en_MRI = model.encoder(MRI_gm)
        # fusion
        
        f_type=fusion_type[0]
        f = model.fusion(en_FDG, en_MRI)
        print(en_FDG[0].shape,en_FDG[1].shape,en_FDG[2].shape,en_FDG[3].shape)
        # decoder
        outputs = model.decoder_train(f)
        
        #outputs=model(FDG_1c,MRI_gm)
        
        

        

        ssim_loss_value = 0.
        pixel_loss_value = 0.
        gradient_loss_value = 0.

        if opt.adaptive:
            pixel_loss_FDG, pixel_loss_MRI = mse_loss(outputs, FDG_1c, MRI_1c)
            ssim_loss_FDG, ssim_loss_MRI = ssim_loss(outputs, FDG_1c, MRI_1c)
        else:
            pixel_loss_FDG = mse_loss(outputs, FDG_1c)*10/ opt.batchSize
            pixel_loss_MRI = mse_loss(outputs, MRI_1c)/ opt.batchSize

            ssim_loss_FDG = (1-pytorch_msssim.msssim(outputs, FDG_1c))/opt.batchSize#/opt.batchSize#, normalize=True
            ssim_loss_MRI = (1-pytorch_msssim.msssim(outputs, MRI_1c))/opt.batchSize#/opt.batchSize#, normalize=True
        if opt.gradient:
            
            gd_loss_FDG=gd_loss(outputs, FDG_1c)
            gd_loss_MRI=gd_loss(outputs, MRI_1c)
        else:
            gd_loss_FDG=0
            gd_loss_MRI=0
        
        
        #pixel_loss_value += 0.8*pixel_loss_FDG#*pixelRatio
        #pixel_loss_value += 0.2*pixel_loss_MRI
        
        pixel_loss_value += pixel_loss_FDG
        pixel_loss_value += pixel_loss_MRI

        ssim_loss_value += ssim_loss_FDG
        ssim_loss_value += ssim_loss_MRI
        
        gradient_loss_value += 0.2*gd_loss_FDG
        gradient_loss_value += 0.8*gd_loss_MRI
        
        # total loss
        #total_loss = pixel_loss_value + ratio_ssim * ssim_loss_value + gradient_loss_value
        # total loss
        if opt.adaptive:
            #---->only FDG SSIM decreasing
            #total_loss = 20*pixel_loss_value + ssim_loss_value + gradient_loss_value
            
            #total_loss = 20*pixel_loss_value + 10*ssim_loss_value + gradient_loss_value
        
            total_loss = pixel_loss_value + 10*ssim_loss_value + gradient_loss_value

        else:

            total_loss = pixel_loss_value + ratio_ssim * ssim_loss_value + gradient_loss_value


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
    
