import torch
import torchvision.datasets as dsets
from torchvision import transforms
from data__set import Data__Set
import torchvision.transforms as trns
import matplotlib.pyplot as plt

class Data__Loader():
    def __init__(self, image_path, image_size, batch_size, shuf=True):
        #self.dataset = dataset
        self.path0 = image_path[0]
        self.path1 = image_path[1]
        self.path2 = image_path[2]
        self.imsize = image_size
        self.batch = batch_size
        self.shuf = shuf

    def loader(self):

        train_transform = trns.Compose([
            #trns.Resize((256, 256)),
            #trns.RandomCrop((224, 224)),
            #trns.RandomHorizontalFlip(),
            #trns.Resize((224, 224)),
            #transforms.Grayscale(num_output_channels=3),
            trns.ToTensor()
            #trns.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        train_set = Data__Set(root0=self.path0,root1=self.path1, root2=self.path2, transform=train_transform)

        loader = torch.utils.data.DataLoader(dataset=train_set,
                                              batch_size=self.batch,
                                              shuffle=self.shuf,
                                              num_workers=2,
                                              drop_last=True)
        return loader#, train_set.path


'''if __name__=='__main__':
    root=[['src\\FDG_IMG\\137_S_4520','src\\FDG_IMG\\018_S_4349'],['src\\MRI_IMG\\137_S_4520','src\\MRI_IMG\\018_S_4349']]# , 'src\\FDG_IMG\\032_S_4921', 'src\\FDG_IMG\\100_S_4469', 'src\\FDG_IMG\\010_S_4345', 'src\\FDG_IMG\\032_S_4386', 'src\\FDG_IMG\\116_S_4453', 'src\\FDG_IMG\\007_S_4488', 'src\\FDG_IMG\\168_S_6065', 'src\\FDG_IMG\\007_S_4620', 'src\\FDG_IMG\\137_S_4632', 'src\\FDG_IMG\\082_S_4428', 'src\\FDG_IMG\\018_S_4400', 'src\\FDG_IMG\\135_S_4446', 'src\\FDG_IMG\\006_S_4357', 'src\\FDG_IMG\\036_S_4491', 'src\\FDG_IMG\\068_S_4424', 'src\\FDG_IMG\\029_S_4585', 'src\\FDG_IMG\\010_S_4442', 'src\\FDG_IMG\\128_S_4609', 'src\\FDG_IMG\\029_S_4385', 'src\\FDG_IMG\\006_S_4485', 'src\\FDG_IMG\\033_S_4505', 'src\\FDG_IMG\\116_S_4483', 'src\\FDG_IMG\\018_S_4399', 'src\\FDG_IMG\\013_S_4579', 'src\\FDG_IMG\\168_S_6059', 'src\\FDG_IMG\\018_S_4313', 'src\\FDG_IMG\\073_S_4795', 'src\\FDG_IMG\\037_S_4308', 'src\\FDG_IMG\\009_S_4388', 'src\\FDG_IMG\\016_S_4952', 'src\\FDG_IMG\\003_S_4872', 'src\\FDG_IMG\\012_S_4545', 'src\\FDG_IMG\\031_S_4474', 'src\\FDG_IMG\\007_S_1206', 'src\\FDG_IMG\\128_S_4599', 'src\\FDG_IMG\\006_S_4449', 'src\\FDG_IMG\\094_S_4459', 'src\\FDG_IMG\\029_S_4652', 'src\\FDG_IMG\\137_S_4482', 'src\\FDG_IMG\\072_S_4391', 'src\\FDG_IMG\\941_S_4376', 'src\\FDG_IMG\\070_S_4856', 'src\\FDG_IMG\\037_S_4410', 'src\\FDG_IMG\\094_S_4649', 'src\\FDG_IMG\\036_S_4389', 'src\\FDG_IMG\\098_S_4506', 'src\\FDG_IMG\\073_S_4762', 'src\\FDG_IMG\\341_S_6653', 'src\\FDG_IMG\\128_S_4607', 'src\\FDG_IMG\\032_S_4429', 'src\\FDG_IMG\\073_S_4552', 'src\\FDG_IMG\\036_S_4878', 'src\\FDG_IMG\\137_S_4587', 'src\\FDG_IMG\\073_S_4393', 'src\\FDG_IMG\\127_S_4604', 'src\\FDG_IMG\\029_S_4279', 'src\\FDG_IMG\\168_S_6062', 'src\\FDG_IMG\\073_S_4559', 'src\\FDG_IMG\\073_S_5023', 'src\\FDG_IMG\\135_S_4598', 'src\\FDG_IMG\\029_S_6505', 'src\\FDG_IMG\\019_S_4835', 'src\\FDG_IMG\\003_S_4555', 'src\\FDG_IMG\\053_S_4578', 'src\\FDG_IMG\\013_S_4616', 'src\\FDG_IMG\\003_S_4644', 'src\\FDG_IMG\\014_S_4577', 'src\\FDG_IMG\\137_S_4466', 'src\\FDG_IMG\\941_S_4292', 'src\\FDG_IMG\\116_S_4855', 'src\\FDG_IMG\\136_S_4433', 'src\\FDG_IMG\\128_S_4832', 'src\\FDG_IMG\\013_S_4580', 'src\\FDG_IMG\\073_S_4739', 'src\\FDG_IMG\\109_S_4499', 'src\\FDG_IMG\\009_S_4612', 'src\\FDG_IMG\\153_S_4372', 'src\\FDG_IMG\\041_S_4427'], ['src\\MRI_IMG\\137_S_4520', 'src\\MRI_IMG\\018_S_4349', 'src\\MRI_IMG\\032_S_4921', 'src\\MRI_IMG\\100_S_4469', 'src\\MRI_IMG\\010_S_4345', 'src\\MRI_IMG\\032_S_4386', 'src\\MRI_IMG\\116_S_4453', 'src\\MRI_IMG\\007_S_4488', 'src\\MRI_IMG\\168_S_6065', 'src\\MRI_IMG\\007_S_4620', 'src\\MRI_IMG\\137_S_4632', 'src\\MRI_IMG\\082_S_4428', 'src\\MRI_IMG\\018_S_4400', 'src\\MRI_IMG\\135_S_4446', 'src\\MRI_IMG\\006_S_4357', 'src\\MRI_IMG\\036_S_4491', 'src\\MRI_IMG\\068_S_4424', 'src\\MRI_IMG\\029_S_4585', 'src\\MRI_IMG\\010_S_4442', 'src\\MRI_IMG\\128_S_4609', 'src\\MRI_IMG\\029_S_4385', 'src\\MRI_IMG\\006_S_4485', 'src\\MRI_IMG\\033_S_4505', 'src\\MRI_IMG\\116_S_4483', 'src\\MRI_IMG\\018_S_4399', 'src\\MRI_IMG\\013_S_4579', 'src\\MRI_IMG\\168_S_6059', 'src\\MRI_IMG\\018_S_4313', 'src\\MRI_IMG\\073_S_4795', 'src\\MRI_IMG\\037_S_4308', 'src\\MRI_IMG\\009_S_4388', 'src\\MRI_IMG\\016_S_4952', 'src\\MRI_IMG\\003_S_4872', 'src\\MRI_IMG\\012_S_4545', 'src\\MRI_IMG\\031_S_4474', 'src\\MRI_IMG\\007_S_1206', 'src\\MRI_IMG\\128_S_4599', 'src\\MRI_IMG\\006_S_4449', 'src\\MRI_IMG\\094_S_4459', 'src\\MRI_IMG\\029_S_4652', 'src\\MRI_IMG\\137_S_4482', 'src\\MRI_IMG\\072_S_4391', 'src\\MRI_IMG\\941_S_4376', 'src\\MRI_IMG\\070_S_4856', 'src\\MRI_IMG\\037_S_4410', 'src\\MRI_IMG\\094_S_4649', 'src\\MRI_IMG\\036_S_4389', 'src\\MRI_IMG\\098_S_4506', 'src\\MRI_IMG\\073_S_4762', 'src\\MRI_IMG\\341_S_6653', 'src\\MRI_IMG\\128_S_4607', 'src\\MRI_IMG\\032_S_4429', 'src\\MRI_IMG\\073_S_4552', 'src\\MRI_IMG\\036_S_4878', 'src\\MRI_IMG\\137_S_4587', 'src\\MRI_IMG\\073_S_4393', 'src\\MRI_IMG\\127_S_4604', 'src\\MRI_IMG\\029_S_4279', 'src\\MRI_IMG\\168_S_6062', 'src\\MRI_IMG\\073_S_4559', 'src\\MRI_IMG\\073_S_5023', 'src\\MRI_IMG\\135_S_4598', 'src\\MRI_IMG\\029_S_6505', 'src\\MRI_IMG\\019_S_4835', 'src\\MRI_IMG\\003_S_4555', 'src\\MRI_IMG\\053_S_4578', 'src\\MRI_IMG\\013_S_4616', 'src\\MRI_IMG\\003_S_4644', 'src\\MRI_IMG\\014_S_4577', 'src\\MRI_IMG\\137_S_4466', 'src\\MRI_IMG\\941_S_4292', 'src\\MRI_IMG\\116_S_4855', 'src\\MRI_IMG\\136_S_4433', 'src\\MRI_IMG\\128_S_4832', 'src\\MRI_IMG\\013_S_4580', 'src\\MRI_IMG\\073_S_4739', 'src\\MRI_IMG\\109_S_4499', 'src\\MRI_IMG\\009_S_4612', 'src\\MRI_IMG\\153_S_4372', 'src\\MRI_IMG\\041_S_4427']]
    print(root[0])
    print(root[1])

    
    dld=Data__Loader(image_path=root, image_size=(109,109), batch_size=16)
    dld=dld.loader()
    for a,d in enumerate(dld):
        print("a ", a)
        print(d.shape)
        
    plt.figure(figsize=(24,8))  
    plt.subplot(1,2,1)
    plt.imshow(d[0,0,:,:], cmap='gray')
    plt.subplot(1,2,2)
    plt.imshow(d[0,1,:,:], cmap='gray')

    plt.show()'''