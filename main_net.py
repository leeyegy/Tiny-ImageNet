import torch.nn as nn
import torch
import  numpy as np
import torchvision
from torchvision import datasets, models,transforms
import  os ,re ,glob
import torch.optim as optim
import h5py
from torch.optim import lr_scheduler
import sys


data_dir = "/data/liyanjie/Tiny-ImageNet/Tiny-ImageNet/tiny-imagenet-200/"
save_dir = 'checkpoint/'
model_name = "resnet"
num_classes = 200
batch_size = 100
num_epochs = 200

# load pretrained model
model = models.resnet50(pretrained=True)
model.avgpool = nn.AdaptiveAvgPool2d(1)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes) # change the last layer for 200-classification
print("model structure of resnet-50")
print (model)

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
    ]),
}
print("Initializing Datasets and DataLoaders...")
image_datasets = {x:datasets.ImageFolder(os.path.join(data_dir,x),data_transforms[x]) for x in ['train','val']}
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x],batch_size=batch_size,shuffle=True,num_workers=64) for x in ['train','val']}


# load device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load("checkpoint/resnet50_epoch_21.pth")
model = model.to(device)

# define optimizer and loss func
optimizer = optim.SGD(model.parameters(),lr=0.001,momentum=0.9)
criterion = nn.CrossEntropyLoss()
# scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)




for epoch in range(22,num_epochs):
    print ('Epoch {}/{}'.format(epoch,num_epochs-1))
    print ('-' * 10)

    for phase in ['train','val']:
        model.train() if phase == 'train' else model.eval()
        # sys.stdout.flush()
        running_loss = 0.0
        running_corrects = 0

        for batch_idx ,[data,target] in enumerate(dataloaders_dict[phase]):
            # print("\r{}/{}".format(batch_idx,len(dataloaders_dict[phase].dataset)/batch_size),end = '')
            optimizer.zero_grad()
            data,target = data.to(device),target.to(device)

            if phase == 'train':
                output = model(data)
            else:
                with torch.no_grad():
                    output = model(data)

            pred =   output.max(1,keepdim=True)[1]
            running_corrects += pred.eq(target.view_as(pred)).sum().item()
            loss = criterion(output,target)
            running_loss += loss.item()
            if phase == 'train':
                loss.backward()
                optimizer.step()
                # scheduler.step()
        print('{} loss: {:.4f} Acc: {}/{} ({:.4f})'.format(phase,running_loss,running_corrects,len(dataloaders_dict[phase].dataset),running_corrects/len(dataloaders_dict[phase].dataset)))
        # sys.stdout.flush()

    torch.save(model,os.path.join('checkpoint',"resnet50_epoch_"+str(epoch)+".pth"))


