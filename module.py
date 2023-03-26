
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import pytorch_msssim

class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()
        self.kernel_x = torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).unsqueeze(0).unsqueeze(0)
        self.kernel_y = torch.FloatTensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).unsqueeze(0).unsqueeze(0)
        self.kernel_x = self.kernel_x.cuda()
        self.kernel_y = self.kernel_y.cuda()
    def forward(self, outputs, targets):
        gx_target = F.conv2d(targets, self.kernel_x, stride=1, padding=1)
        gy_target = F.conv2d(targets, self.kernel_y, stride=1, padding=1)
        gx_output = F.conv2d(outputs, self.kernel_x, stride=1, padding=1)
        gy_output = F.conv2d(outputs, self.kernel_y, stride=1, padding=1)
        grad_loss = F.mse_loss(gx_target, gx_output) + F.mse_loss(gy_target, gy_output)
        return grad_loss#gx_target, gx_output, gy_target, gy_output

def Fro_LOSS(batchimg):
    
    fro_norm = torch.norm(batchimg, p= 'fro',dim=(2,3))/ (int(batchimg.shape[2]) * int(batchimg.shape[3]))
    #torch.mean(fro_norm)
	#fro_norm = tf.square(tf.norm(batchimg, axis = [1, 2], ord = 'fro')) / (int(batchimg.shape[1]) * int(batchimg.shape[2]))\
	#E = tf.reduce_mean(fro_norm)
    return fro_norm
def features_grad(features):
    kernel = torch.FloatTensor([[1 / 8, 1 / 8, 1 / 8], [1 / 8, -1, 1 / 8], [1 / 8, 1 / 8, 1 / 8]])
    kernel = kernel[None,None,:,:].cuda()
    print("features",features.shape)
    c, _, _ = features.shape
    c = int(c)
    for i in range(c):
        feat=features[i, :, :]
        feat=feat[None,None,:,:]
        print("feat", feat.shape)
        fg = F.conv2d(feat, kernel,padding = 1)#, strides = 1,padding = 1)
        if i == 0:
            fgs = fg
        else:
            #fgs = tf.concat([fgs, fg], axis = -1)
            fgs = torch.cat([fgs, fg], axis = -1)
    return fgs




class AdaptiveMSE(nn.Module):
    def __init__(self):
        super(AdaptiveMSE, self).__init__()

    def forward(self, source1, source2, output):
        
        mse1 = Fro_LOSS(output-source1)
        mse2 = Fro_LOSS(output-source2)
        #print("mse1, mse2",mse1, mse2)
        for i in range(len(source1)):
            '''self.m1 = torch.mean(torch.square(features_grad(source1[i])), axis = [1, 2, 3])
            self.m2 = torch.mean(torch.square(features_grad(source2[i])), axis = [1, 2, 3])'''
            self.m1 = torch.mean(torch.square(source1[i]), axis = [1, 2])
            self.m2 = torch.mean(torch.square(source2[i]), axis = [1, 2])

            #print("m1, m2",self.m1, self.m1)
            if i == 0:
                '''self.ws1 = tf.expand_dims(self.m1, axis = -1)
                self.ws2 = tf.expand_dims(self.m2, axis = -1)'''

                self.ws1 = self.m1[:,None]
                self.ws2 = self.m2[:,None]
            else:

                #self.ws1[]
                '''self.ws1 = tf.concat([self.ws1, tf.expand_dims(self.m1, axis = -1)], axis = -1)
                self.ws2 = tf.concat([self.ws2, tf.expand_dims(self.m2, axis = -1)], axis = -1)'''

                self.ws1 = torch.cat([self.ws1,self.m1[:,None]], axis = -1)
                self.ws2 = torch.cat([self.ws2,self.m2[:,None]], axis = -1)
            #print("ws1, ws2",self.ws1,self.ws2)

        self.s1 = torch.mean(self.ws1, axis = -1) / len(source1)
        self.s2 = torch.mean(self.ws2, axis = -1) / len(source2)
        #print("s1",self.s1.shape)
        self.s = F.softmax((torch.cat([self.s1[:,None], self.s2[:,None]], axis = -1)))
        #print("s",self.s.shape)
        #print("s",self.s)

        self.mse_loss1=torch.mean(self.s[:, 0] * mse1)
        self.mse_loss2=torch.mean(self.s[:, 1] * mse2)

        #print("self.mse_loss1,self.mse_loss2",self.mse_loss1,self.mse_loss2)
        #self.ssim_loss = tf.reduce_mean(self.s[:, 0] * SSIM1 + self.s[:, 1] * SSIM2)
        #self.mse_loss = torch.mean(self.s[:, 0] * mse1 + self.s[:, 1] * mse2)
        #self.content_loss = self.ssim_loss + 20 * self.mse_loss
        
        
        '''ratio1=torch.norm(source1,dim=(2,3)).cuda()
        ratio2=torch.norm(source2,dim=(2,3)).cuda()
        
        ratio=torch.cat((ratio1,ratio2),1)
        #avoid divided by zero
        ratio=(ratio-torch.mean(ratio,dim=0))/(torch.std(ratio,dim=0)+0.00001)
        #if the numbers are too big the exponents will probably blow up,
        #to avoid this, first shift the highest value in array to zero.
        MAX=torch.cat((torch.max(ratio,dim=1)[0],torch.max(ratio,dim=1)[0]),0)
        MAX=MAX.view(-1,2)
        softmax_ratio=F.softmax(ratio-MAX,dim=1)
        
        loss_mse = torch.nn.MSELoss(reduction='none')
        mse1 = torch.sum(torch.sum(loss_mse(output, source1) ,dim=3),dim=2)/(source1.shape[2]*source1.shape[3])
        mse2 = torch.sum(torch.sum(loss_mse(output, source2) ,dim=3),dim=2)/(source2.shape[2]*source2.shape[3])
        mse=torch.cat((mse1,mse2),1)'''
        
        #source1_loss,source2_loss=torch.sum(torch.mul(softmax_ratio,mse),dim=0)/source1.shape[0]

        return self.mse_loss1,self.mse_loss2


class AdaptiveSSIM(nn.Module):
    def __init__(self):
        super(AdaptiveSSIM, self).__init__()

    def forward(self, source1, source2, output):
        ssim1 = 1-pytorch_msssim.ssim(output, source1, size_average=False, val_range=1.0)
        ssim2 = 1-pytorch_msssim.ssim(output, source2, size_average=False, val_range=1.0)
        #print("ssim1,ssim2",ssim1,ssim2)
        for i in range(len(source1)):
            self.m1 = torch.mean(torch.square(source1[i]), axis = [1, 2])
            self.m2 = torch.mean(torch.square(source2[i]), axis = [1, 2])

            #print("m1, m2",self.m1, self.m1)
            if i == 0:
                self.ws1 = self.m1[:,None]
                self.ws2 = self.m2[:,None]
            else:
                self.ws1 = torch.cat([self.ws1,self.m1[:,None]], axis = -1)
                self.ws2 = torch.cat([self.ws2,self.m2[:,None]], axis = -1)
            #print("ws1, ws2",self.ws1,self.ws2)

        self.s1 = torch.mean(self.ws1, axis = -1) / len(source1)
        self.s2 = torch.mean(self.ws2, axis = -1) / len(source2)
        #print("s1",self.s1.shape)
        self.s = F.softmax((torch.cat([self.s1[:,None], self.s2[:,None]], axis = -1)))
        #print("s",self.s)

        self.ssim_loss1=torch.mean(self.s[:, 0] * ssim1)
        self.ssim_loss2=torch.mean(self.s[:, 1] * ssim2)
        #print("self.ssim_loss1,self.ssim_loss2",self.ssim_loss1,self.ssim_loss2)
        '''ratio1=torch.norm(source1,dim=(2,3)).cuda()
        ratio2=torch.norm(source2,dim=(2,3)).cuda()
        ratio=torch.cat((ratio1,ratio2),1)
        #avoid divided by zero
        ratio=(ratio-torch.mean(ratio,dim=0))/(torch.std(ratio,dim=0)+0.00001)
        #if the numbers are too big the exponents will probably blow up,
        #to avoid this, first shift the highest value in array to zero.
        MAX=torch.cat((torch.max(ratio,dim=1)[0],torch.max(ratio,dim=1)[0]),0)
        MAX=MAX.view(-1,2)
        softmax_ratio=F.softmax(ratio-MAX,dim=1)
        #print(output.shape)
        #print(source1.shape)
        ssim=torch.cat((ssim1[:,None],ssim2[:,None]),1)
        #sprint(ssim.shape,ssim)
        source1_loss,source2_loss=torch.sum(torch.mul(softmax_ratio,ssim),dim=0)/source1.shape[0]'''
        
        return self.ssim_loss1,self.ssim_loss2    



if __name__ == "__main__":
    '''source1=torch.rand(size=(2,1,11,11)).cuda()
    source2=torch.rand(size=(2,1,11,11)).cuda()
    output=torch.rand(size=(2,1,11,11)).cuda()'''
    
    output=torch.FloatTensor([[[[1,1,1], [0,0,0], [1,1,1]]]]).cuda()
    source1=torch.FloatTensor([[[[0,0,0], [0,0,0], [0,0,0]]]]).cuda()
    source2=torch.FloatTensor([[[[1,1,1], [1,1,1], [1,1,1]]]]).cuda()


    amse=AdaptiveMSE()
    print(amse(source1,source2,output))

    assim=AdaptiveSSIM()
    print(assim(source1,source2,output))


    
'''class AdaptiveMSE(nn.Module):
    def __init__(self):
        super(AdaptiveMSE, self).__init__()

    def forward(self, source1, source2, output):
        ratio1=torch.norm(source1,dim=(2,3)).cuda()
        ratio2=torch.norm(source2,dim=(2,3)).cuda()
        ratio=torch.cat((ratio1,ratio2),1)
        #avoid divided by zero
        ratio=(ratio-torch.mean(ratio,dim=0))/(torch.std(ratio,dim=0)+0.00001)
        #if the numbers are too big the exponents will probably blow up,
        #to avoid this, first shift the highest value in array to zero.
        MAX=torch.cat((torch.max(ratio,dim=1)[0],torch.max(ratio,dim=1)[0]),0)
        MAX=MAX.view(-1,2)
        softmax_ratio=F.softmax(ratio-MAX,dim=1)
        
        loss_mse = torch.nn.MSELoss(reduction='none')
        mse1 = torch.sum(torch.sum(loss_mse(output, source1) ,dim=3),dim=2)/(source1.shape[2]*source1.shape[3])
        mse2 = torch.sum(torch.sum(loss_mse(output, source2) ,dim=3),dim=2)/(source2.shape[2]*source2.shape[3])
        mse=torch.cat((mse1,mse2),1)
        
        source1_loss,source2_loss=torch.sum(torch.mul(softmax_ratio,mse),dim=0)/source1.shape[0]

        return source1_loss,source2_loss'''    
    
'''class AdaptiveSSIM(nn.Module):
    def __init__(self):
        super(AdaptiveSSIM, self).__init__()

    def forward(self, source1, source2, output):
        ratio1=torch.norm(source1,dim=(2,3)).cuda()
        ratio2=torch.norm(source2,dim=(2,3)).cuda()
        ratio=torch.cat((ratio1,ratio2),1)
        #avoid divided by zero
        ratio=(ratio-torch.mean(ratio,dim=0))/(torch.std(ratio,dim=0)+0.00001)
        #if the numbers are too big the exponents will probably blow up,
        #to avoid this, first shift the highest value in array to zero.
        MAX=torch.cat((torch.max(ratio,dim=1)[0],torch.max(ratio,dim=1)[0]),0)
        MAX=MAX.view(-1,2)
        softmax_ratio=F.softmax(ratio-MAX,dim=1)
        #print(output.shape)
        #print(source1.shape)

        ssim1 = 1-pytorch_msssim.ssim(output, source1, size_average=False, val_range=1.0)#/opt.batchSize#, normalize=True
        ssim2 = 1-pytorch_msssim.ssim(output, source2, size_average=False, val_range=1.0)
        
        ssim=torch.cat((ssim1[:,None],ssim2[:,None]),1)
        #sprint(ssim.shape,ssim)
        source1_loss,source2_loss=torch.sum(torch.mul(softmax_ratio,ssim),dim=0)/source1.shape[0]

        return source1_loss,source2_loss    '''

'''    self.kernel_x = torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).unsqueeze(0).unsqueeze(0)
        self.kernel_y = torch.FloatTensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).unsqueeze(0).unsqueeze(0)
        self.kernel_x = self.kernel_x.cuda()
        self.kernel_y = self.kernel_y.cuda()
    def forward(self, outputs, targets):
        gx_target = F.conv2d(targets, self.kernel_x, stride=1, padding=1)'''





class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(
        self, name="accuracy", minmax=0, output_folder=""
    ):
        self.best_epoch=0
        self.minmax = minmax
        if self.minmax==0:
            
            self.best_para = float(100)
        if self.minmax==1:
            
            self.best_para = float(0)
        self.name = name
        self.output_folder = output_folder
        
        
        
    def __call__(
        self, current_para, epoch, fold, model
    ):
        if not os.path.exists(self.output_folder+"checkpoint/"):
            os.makedirs(self.output_folder+"checkpoint/")
        if self.minmax==1:
            if current_para > self.best_para:
                self.best_para = current_para
                
                print("\n------------>Best parameter {} : {}".format(self.name, self.best_para))
                print("\nSaved at epoch: {}\n".format(epoch))

                model_out_path = self.output_folder+"/checkpoint/best_" + str(self.name)+"_fold_" +str(fold)+".pth"
                torch.save(model.state_dict(), model_out_path)
                self.best_epoch=epoch

                
        if self.minmax==0:
            if current_para < self.best_para:
                self.best_para = current_para

                print("\n------------>Best parameter {} : {}".format(self.name, self.best_para))
                print("\nSaved at epoch: {}\n".format(epoch))

                model_out_path = self.output_folder+"/checkpoint/best_" + str(self.name)+"_fold_" +str(fold)+".pth"
                torch.save(model.state_dict(), model_out_path)
                self.best_epoch=epoch