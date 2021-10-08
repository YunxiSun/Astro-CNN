import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import time
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import cv2

#定义参数
EPOCH=100 #训练epoch次数
BATCH_SIZE=3 #批训练的数量
LR=0.0001 #学习率

mask_img_path='./train/set_0/'#定义路径
test_img_path='./train/set_1'
model_path='./model3.pkl'
LOSS=[]

class ToTensor(object):
    def __call__(self, sample):
        image,labels=sample['image'],sample['labeles']
        return {'image':torch.from_numpy(image),
                'labels':torch.from_numpy(labels)}

data_transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
])#数据转换为tensor

def removeneg(im):
    """
    remove negativ value of array
    """
    im2 = np.copy(im)
    arr = np.isnan(im2)
    im2[arr] = 0
    arr2 = np.isinf(im2)
    im2[arr2] = 0
    im2 = np.maximum(im2,0)
    # im2[im1<2] =0
    # im2 = np.minimum(im2,1e12)
    # im2 = signal.medfilt2d(im2,kernel_size=5)

    return im2

#自定义dataset
class SDataset(Dataset):
    def __init__(self,mask_img_path,transform=None):
        self.mask_img_path = mask_img_path
        self.transform = transform
        self.mask_img = os.listdir(mask_img_path)
    def __len__(self):
        return len(self.mask_img)
    def __getitem__(self, idx):
        label_img_name=os.path.join(mask_img_path,"heap_{}.fits".format(idx))
        hdu=fits.open(label_img_name)
        hdu.verify('fix')
        mask_img=hdu[0].data
        image_raw = np.zeros([6,1024, 1024], dtype=np.float64)
        label_img_raw=np.zeros([1024,1024],dtype=np.float64)
        label_img_raw[:, :] = removeneg(mask_img[6].data)
        label_img_raw=np.log(label_img_raw+1)
        for num in range(0,5):
            image_raw[num,:,:]=removeneg(mask_img[num].data)
        image_raw=np.log(image_raw+1)
        sample = {'image': image_raw, 'labels': label_img_raw}
        return sample

#自定义卷积模型

#全卷积函数
def PL(data,label,epoch,loss):
    data=data.cpu()
    label=label.cpu()
    #data=torch.mul(255).byte
    #data=data.numpy()
    data=data.detach().numpy()
    label=label.detach().numpy()

    loss1=label[0]-data[0][0]
    loss2=label[1]-data[1][0]
    loss3=label[2]-data[2][0]

    LOSS.append(loss)
    plt.ion()
    plt.subplot(3, 3, 1)
    plt.imshow(data[0][0],cmap='gray', label='train')
    plt.title('train')
    plt.subplot(3, 3, 2)
    plt.imshow(label[0], cmap='gray', label='label')
    plt.title('label')
    plt.subplot(3, 3, 3)
    plt.imshow(loss1,cmap='gray', label='loss_image')
    plt.title('loss')
    plt.subplot(3, 3, 4)
    plt.imshow(data[1][0], cmap='gray', label='train')
    plt.title('train')
    plt.subplot(3, 3, 5)
    plt.imshow(label[1], cmap='gray', label='label')
    plt.title('label')
    plt.subplot(3, 3, 6)
    plt.imshow(loss2, cmap='gray', label='loss_image')
    plt.title('loss')
    plt.subplot(3, 3, 7)
    plt.imshow(data[2][0], cmap='gray', label='train')
    plt.title('train')
    plt.subplot(3, 3, 8)
    plt.imshow(label[2], cmap='gray', label='label')
    plt.title('label')
    plt.subplot(3, 3, 9)
    plt.imshow(loss3, cmap='gray', label='loss_image')
    plt.title('loss')
    plt.pause(5)

    plt.show()

def PL1(data,label,epoch):
    data = data.cpu()
    label = label.cpu()
    # data=torch.mul(255).byte
    # data=data.numpy()
    data = data.detach().numpy()
    label = label.detach().numpy()

    loss1 = label[0] - data[0][0]
    x1=range(0,epoch+1)
    y1=LOSS
    plt.ion()
    plt.subplot(1, 4, 1)
    plt.imshow(data[0][0], cmap='gray', label='train')
    plt.title('train')
    plt.subplot(1, 4, 2)
    plt.imshow(label[0], cmap='gray', label='label')
    plt.title('label')
    plt.subplot(1, 4, 3)
    plt.imshow(loss1, cmap='gray', label='loss_image')
    plt.title('loss')
    plt.subplot(1,4,4)
    plt.plot(x1,y1,'-')
    plt.ioff()
    plt.show()

class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        self.conv1=nn.Sequential(
            #nn.Conv2d(6,16,5,1,2),
            nn.Conv2d(6,16,3,1,1),
            nn.BatchNorm2d(16)
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(16,32,3,1,1),
            nn.BatchNorm2d(32)
        )
        self.conv3=nn.Sequential(
            nn.Conv2d(32,64,3,1,1),
            nn.BatchNorm2d(64)
        )
        self.conv4=nn.Sequential(
            nn.Conv2d(64,1,3,1,1),
            nn.BatchNorm2d(1)
        )
    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)

        return x




#导入数据
traindata=SDataset(mask_img_path,transform=data_transform)
testdata=SDataset(test_img_path,transform=data_transform)
#print(test1['1'])
train_loder=DataLoader(dataset=traindata,batch_size=BATCH_SIZE,shuffle=True)
test_loder=DataLoader(dataset=testdata,batch_size=1,shuffle=True)

#导入CUDA
device=torch.device("cuda:0")
#model=CNN().to(device)
#if(os.path.exists('model.pkl')):

    #model=torch.load('\model.pkl')
#else:
    #model=FCN().to(device)
#model=torch.load('\model.pkl')
model=FCN()
if(os.path.exists(model_path)):
    print(os.path.exists(model_path))
    model.load_state_dict(torch.load(model_path))
model.eval()
model.to(device)

#损失函数
criterion=nn.L1Loss()

#优化器
optimizer=optim.Adam(model.parameters(),lr=LR)


#训练
for epoch in range(EPOCH):
    start_time=time.time()
    for i,data in enumerate(train_loder):
        inputs=data['image']
        inputs=inputs.type(torch.FloatTensor)
        labels=data['labels']
        labels=labels.type(torch.FloatTensor)
        inputs,labels=inputs.to(device),labels.to(device)
        outputs=model(inputs)
        loss=criterion(outputs,labels)
        LOSS.append(loss)
        if(epoch%10==0):
          PL1(outputs,labels,epoch)
        #清空上一轮梯度
        optimizer.zero_grad()
        #方向传播
        loss.backward()
        #参数更新
        optimizer.step()
    print('epoch{}  loss:{:.4f}  time:{:.4f}'.format(epoch+1,loss.item(),time.time()-start_time))

torch.save(model.state_dict(),'\Yunxi-Python\exam1\model3.pkl')
model.eval()
eval_loss=0
eval_acc=0

#for i,data in enumerate(test_loder):
    #inputs = data['image']
    #inputs = inputs.type(torch.FloatTensor)
    #labels = data['labels']
    #labels = labels.type(torch.FloatTensor)
    #inputs, labels = inputs.to(device), labels.to(device)
    #outputs=model(inputs)
    #PL1(outputs,labels)
    #loss=criterion(outputs,labels)
    #eval_loss+=loss.item()
    #pred=torch.max(outputs,1)[1].cpu().detach().numpy()[0]
    #print(pred)
    #num_correct=(pred==labels).sum().item()
    #eval_acc+=num_correct
    #print('Test Loss:{:.4f},Acc :{:.4f}'.format(eval_loss/(len(testdata)),eval_acc/(len(testdata))))#





