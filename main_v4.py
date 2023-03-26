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

from net import NestFuse_autoencoder
from torch.optim import Adam
import pytorch_msssim
from args_fusion import args
import scipy.io as scio

# Training settings
parser = argparse.ArgumentParser(description="PyTorch SRResNet")
parser.add_argument("--batchSize", type=int, default=16, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=500, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--step", type=int, default=200, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=500")
parser.add_argument("--cuda", action="store_true", help="Use cuda?")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=0, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained model (default: none)")

parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
parser.add_argument("--dtstCh1", default='src/FDG_mask_npy_111_woutblank', type=str, help="root folder: channel 1 of input dataset")
parser.add_argument("--dtstCh2", default='src/MRI_mask_npy_111_woutblank', type=str, help="root folder: channel 2 of input dataset")

parser.add_argument("--trainingtxt", default='training.txt', type=str, help="pretrained model split dataset")
parser.add_argument("--validationtxt", default='validation.txt', type=str, help="pretrained model split dataset")

parser.add_argument("--pixelRatio", type=int, default=4, help="FDG-MRI importance ratio of pixel loss")
parser.add_argument("--ssimRatio", type=int, default=1, help="MRI-FDG importance ratio of ssim loss")
parser.add_argument("--pixel-ssimRatio", type=int, default=5, help="pixel-ssim ratio in total loss")
parser.add_argument("--saveFeatureMap", type=int, default=200, help="save feature map every ? epoch")
parser.add_argument("--maskRatio", type=int, default=100, help="save feature map every ? epoch")

parser.add_argument("--encoder", action="store_true",default=False ,  help="joint(0, Default)/seperate(1) encoder network")
parser.add_argument("--maskMRI", action="store_true",default=False ,  help="full MRI(0, Default)/grey matter only(1)")

global style_weights,content_weight,style_weight,layer_dict, output_folder
output_folder=str(time.localtime()[0])+"-"+str(time.localtime()[1])+"-"+str(time.localtime()[2])+"-"+str(time.localtime()[3])+str(time.localtime()[4])+str(time.localtime()[5])+'/'
print('output folder', output_folder)

"C:\LAB\PROJECT\FDG_MRI\fusion_NestFuse/2022-11-17-172526/checkpoint/model_epoch_15_10.pth"

Loss_pixel_FDG = []
Loss_ssim_FDG = []
Loss_gmmask_FDG = []
Loss_pixel_MRI = []
Loss_ssim_MRI = []
Loss_gmmask_MRI = []
Loss_all = []
Loss_epoch=[]
def psnr(target, ref, scale=None):
    target_data = np.array(target)
    ref_data = np.array(ref)
    diff = ref_data - target_data
    diff = diff.flatten()
    rmse = math.sqrt( np.mean(diff ** 2.) )
    return 20*math.log10(1.0/rmse)

def load_data(path):
    print("=> Using the dataset from pretrained/resume model")
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


def train_get_data():
    print("=> Split new dataset")
    #print("===> Loading datasets")
    FDG_root=opt.dtstCh1
    FDG_dataset_list=os.listdir(FDG_root)
    for n in range(len(FDG_dataset_list)):
        FDG_dataset_list[n]=os.path.join(FDG_root,FDG_dataset_list[n])     

    MRI_root=opt.dtstCh2
    MRI_dataset_list=os.listdir(MRI_root)
    for n in range(len(MRI_dataset_list)):
        MRI_dataset_list[n]=os.path.join(MRI_root,MRI_dataset_list[n])  
        


    FDG_train, FDG_eval, MRI_train, MRI_eval = train_test_split(FDG_dataset_list, MRI_dataset_list, test_size=0.2, random_state=42)
    FDG_valid, FDG_test, MRI_valid, MRI_test = train_test_split(FDG_eval, MRI_eval, test_size=0.5, random_state=42)
    
    print('FDG_train: ',len(FDG_train))
    print('MRI_train: ',len(MRI_train))
    print('FDG_valid: ',len(FDG_valid))
    print('MRI_test: ',len(MRI_valid))
    print('FDG_test: ',len(FDG_test))
    print('MRI_test: ',len(MRI_test))
    
    if not os.path.exists(output_folder+'split_dataset'):
        os.makedirs(output_folder+'split_dataset')
    path = output_folder+'split_dataset/training.txt'
    f = open(path, 'w')
    f.write("\n".join(FDG_train))
    f.write("\n")
    f.write("\n".join(MRI_train))
    f.close()
    
    path = output_folder+'split_dataset/validation.txt'
    f = open(path, 'w')
    f.write("\n".join(FDG_valid))
    f.write("\n")
    f.write("\n".join(MRI_valid))
    f.close()
    
    path = output_folder+'split_dataset/testing.txt'
    f = open(path, 'w')
    f.write("\n".join(FDG_test))
    f.write("\n")
    f.write("\n".join(MRI_test))
    f.close()

    return FDG_train,MRI_train,FDG_valid,MRI_valid,FDG_test,MRI_test


    
def load_model(path, deepsupervision):
	input_nc = 1
	output_nc = 1
	nb_filter = [64, 112, 160, 208, 256]

	nest_model = NestFuse_autoencoder(nb_filter, input_nc, output_nc, deepsupervision)
	nest_model.load_state_dict(torch.load(path))

	para = sum([np.prod(list(p.size())) for p in nest_model.parameters()])
	type_size = 4
	print('Model {} : params: {:4f}M'.format(nest_model._get_name(), para * type_size / 1000 / 1000))

	nest_model.eval()
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
    if opt.pretrained or opt.resume:
        FDG_train,MRI_train=load_data(opt.trainingtxt)
        FDG_valid,MRI_valid=load_data(opt.validationtxt)
    else:
        FDG_train,MRI_train,FDG_valid,MRI_valid,FDG_test,MRI_test=train_get_data()
    MRI_mask=[]
    for i in range(len(MRI_train)):
        p=MRI_train[i][0:8]+"onlymask7/"+MRI_train[i][31:45]
        MRI_mask.append(p)
        # print(p)
        
    train_path=[FDG_train, MRI_train, MRI_mask]
    valid_path=[FDG_valid, MRI_valid, MRI_mask]
    
    
    #print(str(MRI_train[0][0:8]+"onlymask/"+MRI_train[0][31:45]))
    
    train_dataloader=Data__Loader(image_path=train_path, image_size=(111,111), batch_size=opt.batchSize)
    valid_dataloader=Data__Loader(image_path=valid_path, image_size=(111,111), batch_size=opt.batchSize)




    print("===> Building model")
    input_nc = 1
    output_nc = 1
    nb_filter = [64, 112, 160, 208, 256]
    deepsupervision = False
    nest_model = NestFuse_autoencoder(nb_filter, input_nc, output_nc, deepsupervision)

    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            '''checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint["epoch"] + 1
            nest_model.load_state_dict(checkpoint["model"].state_dict())'''

            nest_model.load_state_dict(torch.load(opt.resume))
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("=> loading model '{}'".format(opt.pretrained))
            '''weights = torch.load(opt.pretrained)
            nest_model.load_state_dict(weights['model'].state_dict())'''
            
            

        else:
            print("=> no model found at '{}'".format(opt.pretrained))


    para = sum([np.prod(list(p.size())) for p in nest_model.parameters()])
    type_size = 4
    print('Model {} : params: {:4f}M'.format(nest_model._get_name(), para * type_size / 1000 / 1000))


    optimizer = Adam(nest_model.parameters(), opt.lr)
    print("===> Training")
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        for i in range(1,2):
            weights_pixel_ssim=opt.pixel_ssimRatio
            weights_mask=opt.maskRatio
            #args.ssim_weight[i]
            train(weights_pixel_ssim, weights_mask, train_dataloader, optimizer, nest_model, epoch)
            save_checkpoint(nest_model, epoch, args.ssim_weight[i])

        '''if epoch%2==1:
            plot_record(epoch)'''

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10"""
    lr = opt.lr * (0.1 ** (epoch // opt.step))
    return lr 

def record_loss_curve(x,y,title):
    plt.figure()
    plt.plot(x,y)
    plt.title(title)
    filename=title+".png"
    filefolder=output_folder+"loss_curve/"
    
    if not os.path.exists(filefolder):
        os.makedirs(filefolder)
    plt.savefig(os.path.join(filefolder,filename))
    plt.close()
    
    
def train(weights_pixel_ssim, weights_mask, train_dataloader, optimizer, model, epoch):

    training_data_loader=train_dataloader.loader()

        
    lr = adjust_learning_rate(optimizer, epoch-1)
    
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    print("Epoch={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))
    

    model.train()


    all_pixel_loss_FDG = 0.
    all_pixel_loss_MRI = 0.
    all_ssim_loss_FDG = 0.
    all_ssim_loss_MRI = 0.
    all_gmmask_loss_FDG = 0.
    all_gmmask_loss_MRI = 0.
    
    all_total_loss = 0.
    mse_loss = torch.nn.MSELoss()
    ssim_loss = pytorch_msssim.msssim
    #nest_model.eval()

    
    
    fusion_type = ['attention_avg', 'attention_max', 'attention_nuclear']

    for iteration, batch in enumerate(training_data_loader, 1):
        

        '''FDG_6c = Variable(batch[:,0,1:7,:,:])
        MRI_6c = Variable(batch[:,1,1:7,:,:])'''
        
        FDG_1c = Variable(batch[:,0,0,:,:],requires_grad=True)
        FDG_1c = FDG_1c[:,None,:,:]
        if opt.maskMRI:
            MRI_1c = Variable(batch[:,1,1,:,:], requires_grad=True)
        else:
            MRI_1c = Variable(batch[:,1,0,:,:], requires_grad=True)
        MRI_1c = MRI_1c[:,None,:,:]
        #print(batch[:,2,:,:,:].shape)
        mask_1c = Variable(batch[:,2,0,:,:],requires_grad=False)
        mask_1c = mask_1c[:,None,:,:]
        
        
        

        if opt.cuda:
            model.cuda()
            '''FDG_6c = FDG_6c.cuda()
            MRI_6c = MRI_6c.cuda()'''
            FDG_1c = FDG_1c.cuda()
            MRI_1c = MRI_1c.cuda()
            mask_1c =  mask_1c.cuda()

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
        f = model.fusion(en_FDG, en_MRI, f_type)
        # decoder
        outputs = model.decoder_train(f)
        #print(outputs.type)
        

    #-----------model


        #x = Variable(FDG_1c.clone(), requires_grad=False)


        FDG_1c_cpu=FDG_1c.cpu().detach()
        outputs_cpu=outputs.cpu().detach()
        MRI_1c_cpu=MRI_1c.cpu().detach() 
        
        FDG_1c_mask_cpu=(FDG_1c*mask_1c).cpu().detach()
        outputs_mask_cpu=(outputs*mask_1c).cpu().detach()
        MRI_1c_mask_cpu=(MRI_1c*mask_1c).cpu().detach() 
        

        ssim_loss_value = 0.
        pixel_loss_value = 0.
        gmmask_loss_value = 0.
        pixelRatio=10#int(opt.pixelRatio)
        ssimRatio=1#int(opt.ssimRatio)
        pixel_loss_FDG = mse_loss(outputs, FDG_1c)* pixelRatio/ len(outputs_cpu)
        pixel_loss_MRI = mse_loss(outputs, MRI_1c)/ len(outputs_cpu)
        ssim_loss_FDG = (1-ssim_loss(outputs, FDG_1c, normalize=True))/ len(outputs_cpu)
        ssim_loss_MRI = (1-ssim_loss(outputs, MRI_1c, normalize=True)) *ssimRatio/ len(outputs_cpu)
        
        gmmask_loss_FDG = mse_loss(outputs_mask_cpu, FDG_1c_mask_cpu)/ len(outputs_cpu)
        gmmask_loss_MRI = mse_loss(outputs_mask_cpu, MRI_1c_mask_cpu)/ len(outputs_cpu)
        
        pixel_loss_value += pixel_loss_FDG
        pixel_loss_value += pixel_loss_MRI

        ssim_loss_value += ssim_loss_FDG
        ssim_loss_value += ssim_loss_MRI
            
        gmmask_loss_value += gmmask_loss_FDG
        gmmask_loss_value += gmmask_loss_MRI
        
        # total loss
        total_loss = pixel_loss_value + weights_pixel_ssim * ssim_loss_value + gmmask_loss_value* weights_mask
        '''total_loss = mse_loss(outputs, FDG_1c)/ len(outputs_cpu) + mse_loss(outputs, MRI_1c)/ len(outputs_cpu) + args.ssim_weight[i] * (1-ssim_loss(outputs, FDG_1c, normalize=True))/ len(outputs_cpu) + args.ssim_weight[i] * (1-ssim_loss(outputs, MRI_1c, normalize=True))/ len(outputs_cpu)'''
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        


        all_pixel_loss_FDG += pixel_loss_FDG.item()
        all_pixel_loss_MRI += pixel_loss_MRI.item()
        all_ssim_loss_FDG += ssim_loss_FDG.item()
        all_ssim_loss_MRI += ssim_loss_MRI.item()
        all_gmmask_loss_FDG += gmmask_loss_FDG.item()
        all_gmmask_loss_MRI += gmmask_loss_MRI.item()
        all_total_loss += total_loss.item()

        if (iteration + 1) % args.log_interval == 0:
            '''mesg = "{}\t SSIM weight {}\tEpoch {}:\t[{}/{}]\t pixel loss: {:.6f}\t ssim loss: {:.6f}\t total: {:.6f}".format(
                time.ctime(), i, epoch, iteration, len(training_data_loader),
                                all_pixel_loss / args.log_interval,
                                (args.ssim_weight[i] * all_ssim_loss) / args.log_interval,
                                (args.ssim_weight[i] * all_ssim_loss + all_pixel_loss) / args.log_interval
            )
            print(mesg)'''
            print("Epoch {}:\t[{}/{}]\t pixel loss FDG: {:.6f}, pixel loss MRI: {:.6f}, ssim loss FDG: {:.6f}, ssim loss MRI: {:.6f},  gmmask loss FDG: {:.6f},  gmmask loss MRI: {:.6f}, pixel: {:.6f}, ssim: {:.6f},  gmmask: {:.6f}, total: {:.6f}\n".format(epoch, iteration, len(training_data_loader), pixel_loss_FDG, pixel_loss_MRI, ssim_loss_FDG, ssim_loss_MRI, gmmask_loss_FDG, gmmask_loss_MRI, pixel_loss_value.item(), ssim_loss_value.item(),  gmmask_loss_value.item(), total_loss.item()))
            
        
    
        '''if (iteration + 1) % (2 * args.log_interval) == 0:
            # save model
            model.eval()
            model.cpu()'''
              
        #print(outputs_cpu.shape)
        if iteration%int(opt.saveFeatureMap)==0:
            plt.figure(figsize=(80,60))  
            for j in range(4):
                
                #for i in range(2):
                plt.subplot(3,4,j+1)
                plt.title("outputs_"+str(j))
                plt.axis('off')
                plt.imshow(outputs_cpu[j,0,:,:], cmap='gray')
                
                plt.subplot(3,4,j+5)
                plt.title("MRI_"+str(j))
                plt.axis('off')
                plt.imshow(MRI_1c_cpu[j,0,:,:], cmap='gray')
                
                plt.subplot(3,4,j+9)
                plt.title("FDG_"+str(j))
                plt.axis('off')
                plt.imshow(FDG_1c_cpu[j,0,:,:], cmap='gray')
            
            filename="epoch"+str(epoch)+"_"+str(iteration)+".png"
            filefolder=output_folder+"output_map/"
            
            if not os.path.exists(filefolder):
                os.makedirs(filefolder)
            plt.savefig(os.path.join(filefolder,filename))
            plt.close()
        #break
            
    Loss_pixel_FDG.append(all_pixel_loss_FDG / len(training_data_loader))
    Loss_ssim_FDG.append(all_ssim_loss_FDG / len(training_data_loader))
    Loss_gmmask_FDG.append(all_gmmask_loss_FDG / len(training_data_loader))
    Loss_pixel_MRI.append(all_pixel_loss_MRI / len(training_data_loader))
    Loss_ssim_MRI.append(all_ssim_loss_MRI / len(training_data_loader))
    Loss_gmmask_MRI.append(all_gmmask_loss_MRI / len(training_data_loader))
    Loss_all.append(all_total_loss / len(training_data_loader)) 
    Loss_epoch.append(int(epoch))
    

    
    #count_loss = count_loss + 1
    all_pixel_loss_FDG = 0.
    all_pixel_loss_MRI = 0.
    all_ssim_loss_FDG = 0.
    all_ssim_loss_MRI = 0. 
    all_gmmask_loss_FDG = 0.
    all_gmmask_loss_MRI = 0. 
    all_total_loss = 0.
        
    if epoch%5==0:
            
        record_loss_curve(Loss_epoch,Loss_pixel_FDG, "FDG_pixel_loss_epoch_"+str(epoch))
        record_loss_curve(Loss_epoch,Loss_ssim_FDG, "FDG_ssim_loss_epoch_"+str(epoch))
        record_loss_curve(Loss_epoch,Loss_pixel_MRI, "MRI_pixel_loss_epoch_"+str(epoch))
        record_loss_curve(Loss_epoch,Loss_ssim_MRI, "MRI_ssim_loss_epoch_"+str(epoch))
        record_loss_curve(Loss_epoch,Loss_gmmask_FDG, "FDG_gmmask_loss_epoch_"+str(epoch))
        record_loss_curve(Loss_epoch,Loss_gmmask_MRI, "MRI_gmmask_loss_epoch_"+str(epoch))
        record_loss_curve(Loss_epoch,Loss_all, "total_loss_epoch_"+str(epoch))    
        
        
        '''#output
        outputs_cpu=outputs.cpu().detach()
        plt.figure()  
        for j in range(16):
            
            #for i in range(2):
            plt.subplot(4,4,j+1)
            plt.title(str(j))
            plt.axis('off')
            plt.imshow(outputs_cpu[j,0,:,:], cmap='gray')
        
        filename="output_epoch"+str(epoch)+"_"+str(iteration)+"_output.png"
        filefolder=output_folder+"output_map/"
        
        if not os.path.exists(filefolder):
            os.makedirs(filefolder)
        plt.savefig(os.path.join(filefolder,filename))
        plt.close()
        
        #MRI
        plt.figure()  
        for j in range(16):
            
            #for i in range(2):
            plt.subplot(4,4,j+1)
            plt.title(str(j))
            plt.axis('off')
            plt.imshow(MRI_1c_cpu[j,0,:,:], cmap='gray')
        
        filename="MRI_epoch"+str(epoch)+"_"+str(iteration)+"_MRI.png"
        filefolder=output_folder+"MRI_map/"
        
        if not os.path.exists(filefolder):
            os.makedirs(filefolder)
        plt.savefig(os.path.join(filefolder,filename))
        plt.close()
        
        #FDG
        plt.figure()  
        for j in range(16):
            
            #for i in range(2):
            plt.subplot(4,4,j+1)
            plt.title(str(j))
            plt.axis('off')
            plt.imshow(FDG_1c_cpu[j,0,:,:], cmap='gray')
        
        filename="epoch"+str(epoch)+"_"+str(iteration)+"_FDG.png"
        filefolder=output_folder+"FDG_map/"
        
        if not os.path.exists(filefolder):
            os.makedirs(filefolder)
        plt.savefig(os.path.join(filefolder,filename))
        plt.close()'''
        
        
        #break        
        '''model_path="model"
        if not os.path.exists(model_path):
            os.makedirs(model_path)
            
            
        save_model_filename =  "Epoch_" + str(epoch) + "_iters_" + str(iteration) + "_" + \
                                str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + args.ssim_path[
                                    i] + ".model"
        save_model_path = os.path.join(model_path, args.ssim_path[i],save_model_filename)
        if not os.path.exists(save_model_path):
            os.makedirs(save_model_path)
        torch.save(model.state_dict(), save_model_path)'''
        
        '''# save loss data
        # pixel loss
        loss_data_pixel = Loss_pixel
        loss_filename_path = args.save_loss_dir + args.ssim_path[i] + '/' + "loss_pixel_epoch_" + str(
            args.epochs) + "_iters_" + str(iteration) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + \
                                args.ssim_path[i] + ".mat"
        scio.savemat(loss_filename_path, {'loss_pixel': loss_data_pixel})
        # SSIM loss
        loss_data_ssim = Loss_ssim
        loss_filename_path = args.save_loss_dir + args.ssim_path[i] + '/' + "loss_ssim_epoch_" + str(
            args.epochs) + "_iters_" + str(iteration) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + \
                                args.ssim_path[i] + ".mat"
        scio.savemat(loss_filename_path, {'loss_ssim': loss_data_ssim})
        # all loss
        loss_data = Loss_all
        loss_filename_path = args.save_loss_dir + args.ssim_path[i] + '/' + "loss_all_epoch_" + str(epoch) + "_iters_" + \
                                str(iteration) + "-" + str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + \
                                args.ssim_path[i] + ".mat"
        scio.savemat(loss_filename_path, {'loss_all': loss_data})'''
        
        '''nest_model.train()
        nest_model.cuda()
        tbar.set_description("\nCheckpoint, trained model saved at", save_model_path)'''

def save_checkpoint(model, epoch, weight):
    model_out_path = output_folder+"checkpoint/" + "model_epoch_{}_{}.pth".format(epoch, weight)
    state = {"epoch": epoch ,"model": model}
    if not os.path.exists(output_folder+"checkpoint/"):
        os.makedirs(output_folder+"checkpoint/")

    torch.save(model.state_dict(), model_out_path)#state

    print("Checkpoint saved to {}".format(model_out_path))
if __name__ == "__main__":
    main()

# pixel loss
'''loss_data_pixel = Loss_pixel
loss_filename_path = args.save_loss_dir + args.ssim_path[i] + '/' + "Final_loss_pixel_epoch_" + str(
    args.epochs) + "_" + str(
    time.ctime()).replace(' ', '_').replace(':', '_') + "_" + args.ssim_path[i] + ".mat"
scio.savemat(loss_filename_path, {'final_loss_pixel': loss_data_pixel})
loss_data_ssim = Loss_ssim
loss_filename_path = args.save_loss_dir + args.ssim_path[i] + '/' + "Final_loss_ssim_epoch_" + str(
    args.epochs) + "_" + str(
    time.ctime()).replace(' ', '_').replace(':', '_') + "_" + args.ssim_path[i] + ".mat"
scio.savemat(loss_filename_path, {'final_loss_ssim': loss_data_ssim})
# SSIM loss
loss_data = Loss_all
loss_filename_path = args.save_loss_dir + args.ssim_path[i] + '/' + "Final_loss_all_epoch_" + str(
    args.epochs) + "_" + str(
    time.ctime()).replace(' ', '_').replace(':', '_') + "_" + args.ssim_path[i] + ".mat"
scio.savemat(loss_filename_path, {'final_loss_all': loss_data})
# save model
nest_model.eval()
nest_model.cpu()
save_model_filename = args.ssim_path[i] + '/' "Final_epoch_" + str(args.epochs) + "_" + \
                        str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + args.ssim_path[i] + ".model"
save_model_path = os.path.join(args.save_model_dir_autoencoder, save_model_filename)
torch.save(nest_model.state_dict(), save_model_path)

print("\nDone, trained model saved at", save_model_path)'''


        

'''if iteration%1000 == 1:    


FDG_1c_cpu=FDG_1c.cpu().detach().numpy()
output_1c_cpu=output_1c.cpu().detach().numpy()
MRI_1c_cpu=MRI_1c.cpu().detach().numpy() 


plt.figure(figsize=(64,32))  
for i in range(6):
    plt.subplot(3,6,i+1) 
    plt.axis('off')
    plt.title('FDG_1c '+str(i))
    plt.imshow(FDG_1c_cpu[i,0,:,:], cmap='gray')
    
    plt.subplot(3,6,i+7)
    plt.axis('off')
    plt.title('output_1c '+str(i))
    plt.imshow(output_1c_cpu[i,0,:,:], cmap='gray')
    
    plt.subplot(3,6,i+13)
    plt.axis('off')
    plt.title('MRI_1c '+str(i))
    plt.imshow(MRI_1c_cpu[i,0,:,:], cmap='gray')
    
    
filename="epoch"+str(epoch)+"_"+str(iteration)+".png"
filefolder=output_folder+"output_train/"
if not os.path.exists(filefolder):
    os.makedirs(filefolder)
plt.savefig(os.path.join(filefolder,filename))
plt.close()   

if opt.vgg_loss:

    content_output_1c_cpu=content_output_1c.cpu().detach().numpy()
    content_MRI_1c_cpu = content_MRI_1c.cpu().detach().numpy()

    
    plt.figure(figsize=(60,20))  
    for i in range(6):
        plt.subplot(2,6,i+1) 
        plt.axis('off')
        plt.title('output_1c '+str(i))
        plt.imshow(content_output_1c_cpu[i,0,:,:], cmap='gray')
    
        plt.subplot(2,6,i+7)
        plt.axis('off')
        plt.title('MRI_1c '+str(i))
        plt.imshow(content_MRI_1c_cpu[i,0,:,:], cmap='gray')
        
    filename="epoch"+str(epoch)+"_"+str(iteration)+".png"
    filefolder=output_folder+"output_vgg/"
    if not os.path.exists(filefolder):
        os.makedirs(filefolder)
    plt.savefig(os.path.join(filefolder,filename))
    plt.close()  '''
