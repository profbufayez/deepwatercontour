import numpy as np
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

def weighted_bce_loss(output,y,weight):
    epsilon= 1e-7
    output = torch.clamp(output, epsilon, 1.-epsilon)
    logit_output = torch.log(output/(1.-output))
    
    loss = (1.-y)*logit_output + (1.+(weight-1.)*y) * (torch.log(1.+torch.exp(-torch.abs(logit_output))) + torch.maximum(-logit_output,torch.tensor(0.).cuda()))
    return torch.sum(loss)/torch.sum(weight)

def weighted_dice_loss(output,y,weight):
    smooth = 1.
    w,m1,m2 = weight*weight, y, output
    intersection = (m1*m2)
    score = (2.*torch.sum(w*intersection)+smooth)/(torch.sum(w*m1)+torch.sum(w*m2)+smooth)
    loss  = 1.-torch.sum(score)
    return loss

def iou_loss(y_pred, y_true, weight):
    weight = weight*weight
    intersection = y_true * y_pred
    not_true     = 1 - y_true
    union        = y_true + (not_true * y_pred)
    iou          = (torch.sum(intersection * weight)) / (torch.sum(union * weight))

    loss = 1-iou
    return loss

def kmeansLoss(output, y,beta=10):
    
    # calculations with y is only to find index
    yDataloader = y.cpu().detach().numpy()
    cix  = np.where(yDataloader == 1)
    ncix = np.where(yDataloader == 0)
    
    contours = output[cix[0],:,cix[1],cix[2]]
    noncontours = output[ncix[0],:,ncix[1],ncix[2]]
    
    muc = torch.mean(contours,axis=0)
    muc = muc.reshape(1,muc.shape[0])
    munc = torch.mean(noncontours,axis=0).reshape(1,muc.shape[1])
    stdc = torch.std(contours,axis=0).reshape(1,muc.shape[1])
        
    loss = torch.mean(stdc *(1.+1/(.1+(torch.cdist(muc,munc)))))
    #    loss = torch.sqrt(torch.sum(stdc*stdc)) *(1.+1/(.1+(torch.cdist(muc,munc))))

    
    return loss/beta
    

def border_loss(output,y,layer,pool_size=(9,9), pad=(4,4)):
    y      = y.type(torch.float32)
    
    output = output.type(torch.float32)
    
    averaged_mask = F.avg_pool2d(y,kernel_size=pool_size,stride=(1,1), padding=pad)
    border = (averaged_mask>0.005).type(torch.float32) * (averaged_mask<0.995).type(torch.float32)
    weight = torch.ones_like(averaged_mask)
    w0     = torch.sum(weight)
    weight+= border*2
    w1     = torch.sum(weight)
    weight*= (w0/w1)
    loss   = kmeansLoss(layer,y,50) + weighted_bce_loss(output,y,weight) + weighted_dice_loss(output,y,weight) + iou_loss(output,y,weight)
    return loss


class NDWIDataset(Dataset):

    def __init__(self, images, masks, transform=None, test_transform=None):
        self.images     = images
        self.masks      = masks
        self.transforms = transform

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        
        image = self.images[idx]
        mask = self.masks[idx]

        if self.transforms:
            augmentations = self.transforms(image=image, mask=mask)
        
        image = augmentations['image']
        mask  = augmentations['mask']
        mask  = mask[np.newaxis,:,:]
        
        return [image.type(torch.float32), 
                mask.type(torch.float32)]


def data(trans, trans_test, X_train, Y_train, X_val, Y_val, X_test, Y_test, split=0.9, val=True, batch_size=16):
    torch.manual_seed(49)
    random.seed(49)
    trainset= NDWIDataset(X_train, Y_train, transform=trans)

    if val:
        print(f'Training:{len(X_train)}, Validation:{len(X_val)}')
        print(f'Testing: {len(X_test)}')
        
        valset  = NDWIDataset(X_val, Y_val, transform=trans_test)
        testset = NDWIDataset(X_test, Y_test, transform=trans_test)
        image_datasets = {'train': trainset, 'val': valset, 'test': testset}
        batch_size = batch_size

        dataloaders = {
          'train': DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory = True),#, num_workers=8),
          'val': DataLoader(valset, batch_size=batch_size, shuffle=True, pin_memory = True),#, num_workers=8),
          'test': DataLoader(testset, batch_size=batch_size, shuffle=False, pin_memory = True)#, num_workers=8)
        }
        
    else:
        print(f'Training:{len(X_train)}')
        print(f'Testing: {len(X_test)}')
        testset = NDWIDataset(X_test, Y_test, transform=trans_test)
        image_datasets = {'train': trainset, 'test': testset}
        batch_size = batch_size

        dataloaders = {
          'train': DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory = True),#, num_workers=8),
          'test': DataLoader(testset, batch_size=batch_size, shuffle=False, pin_memory = True)#, num_workers=8)
        }
        
    
    return dataloaders


def f1_score(y_pred, y_true, threshold=0.5):
    
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)
    y_pred = ((y_pred>threshold)*1.).type(torch.float32)
    
    tp = torch.sum(y_true*(y_pred))
    tn = torch.sum((1-y_true)*((1-y_pred)))
    fp = torch.sum((1-y_true)*(y_pred))
    fn = torch.sum((y_true)*((1-y_pred)))
    
    pr  = ((tp+1.)/(tp+fp+1.))
    rec = ((tp+1.)/(tp+fn+1.))
    f1  = ((2*pr*rec)/(pr+rec))
    return f1


class basic_block(nn.Module):
    def __init__(self,in_channels,out_chan, random_state=0):
        super(basic_block,self).__init__()
        
        #out_channels = out_chan
        out_channels = out_chan//2
        
        torch.manual_seed(random_state)
        self.bn1         = nn.BatchNorm2d(in_channels)
        
        self.conv1x1_1_1 = nn.Conv2d(in_channels, out_channels, 1)
        self.bn2         = nn.BatchNorm2d(out_channels)
        
        self.conv1x1_1_3 = nn.Conv2d(in_channels, out_channels, 1)
        self.bn3         = nn.BatchNorm2d(out_channels)
        
        self.conv3x3_1   = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn4         = nn.BatchNorm2d(out_channels)
        
        self.conv1x1_1_5 = nn.Conv2d(in_channels, out_channels, 1)
        self.bn5         = nn.BatchNorm2d(out_channels)
        
        self.conv5x5_1   = nn.Conv2d(out_channels,out_channels,5,padding=2)
        self.bn6         = nn.BatchNorm2d(out_channels)
        
        self.conv1x1_1_7 = nn.Conv2d(in_channels, out_channels, 1)
        self.bn7         = nn.BatchNorm2d(out_channels)
        
        self.conv7x7_1   = nn.Conv2d(out_channels,out_channels,7,padding=3)
        self.bn8         = nn.BatchNorm2d(out_channels)
        
        self.bn9         = nn.BatchNorm2d(out_channels*4)
        self.conv1x1_2_1 = nn.Conv2d(out_channels*4, out_channels, 1)
        self.bn10        = nn.BatchNorm2d(out_channels)
        self.conv1x1_2_3 = nn.Conv2d(out_channels*4, out_channels, 1)
        self.bn11        = nn.BatchNorm2d(out_channels)
        self.conv3x3_2   = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn12        = nn.BatchNorm2d(out_channels)
        self.conv1x1_2_5 = nn.Conv2d(out_channels*4, out_channels, 1)
        self.bn13        = nn.BatchNorm2d(out_channels)
        self.conv5x5_2   = nn.Conv2d(out_channels,out_channels,5,padding=2)
        self.bn14        = nn.BatchNorm2d(out_channels)
        self.conv1x1_2_7 = nn.Conv2d(out_channels*4, out_channels, 1)
        self.bn15        = nn.BatchNorm2d(out_channels)
        self.conv7x7_2   = nn.Conv2d(out_channels,out_channels,7,padding=3)
        self.bn16        = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        
        bn1         = self.bn1(x)
        conv1x1_1_3 = self.bn3(self.conv1x1_1_3(bn1))
        conv1x1_1_5 = self.bn4(self.conv1x1_1_5(bn1))
        conv1x1_1_7 = self.bn5(self.conv1x1_1_7(bn1))
        conv1x1_1_1 = F.relu(self.bn2(self.conv1x1_1_1(bn1)))
        conv3x3_1   = F.relu(self.bn6(self.conv3x3_1(conv1x1_1_3)))
        conv5x5_1   = F.relu(self.bn7(self.conv5x5_1(conv1x1_1_5)))
        conv7x7_1   = F.relu(self.bn8(self.conv7x7_1(conv1x1_1_7)))
        cat1        = torch.cat([conv1x1_1_1,conv3x3_1,conv5x5_1,conv7x7_1],dim=1)
        
        bn9         = self.bn9(cat1)
        conv1x1_2_3 = self.bn11(self.conv1x1_2_3(bn9))
        conv1x1_2_5 = self.bn12(self.conv1x1_2_5(bn9))
        conv1x1_2_7 = self.bn13(self.conv1x1_2_7(bn9))
        conv1x1_2_1 = F.relu(self.bn10(self.conv1x1_2_1(bn9)))
        conv3x3_2   = F.relu(self.bn14(self.conv3x3_2(conv1x1_2_3)))
        conv5x5_2   = F.relu(self.bn15(self.conv5x5_2(conv1x1_2_5)))
        conv7x7_2   = F.relu(self.bn16(self.conv7x7_2(conv1x1_2_7)))
        cat2        = torch.cat([conv1x1_2_1,conv3x3_2,conv5x5_2,conv7x7_2],dim=1)

  
        return cat2
        

class UNET_multiscale2(nn.Module):
    def __init__(self, in_channels=4, out_channels= 1, init_features=32, random_state=0):
        super(UNET_multiscale2, self).__init__()
        torch.manual_seed(random_state)
        features = init_features
        self.layer1 =  basic_block(in_channels,features)
        self.down1  = nn.Conv2d(features*2,features,2,stride=2)
        
        self.layer2 = basic_block(features,features) 
        self.down2  = nn.Conv2d(features*2,features*2,2,stride=2)

        self.layer3 = basic_block(features*2,features*2) 
        self.down3  = nn.Conv2d(features*4,features*4,2,stride=2)

        self.layer4 = basic_block(features*4,features*4)
        self.down4  = nn.Conv2d(features*8,features*8,2,stride=2)
        
        self.bottleneck = basic_block(features*8,features*8)
        self.bn6     = nn.BatchNorm2d(features*8*2)
        self.up1     = nn.ConvTranspose2d(features*16, features*8, 2, stride=2)
               
        self.layer6  = basic_block(features*16,features*4)      
        self.bn7     = nn.BatchNorm2d(features*4*2)
        self.up2     = nn.ConvTranspose2d(features*8, features*4, 2, stride=2)

        self.layer7  = basic_block(features*8,features*2) 
        self.bn8     = nn.BatchNorm2d(features*2*2)
        self.up3     = nn.ConvTranspose2d(features*4, features*2, 2, stride=2)   
        
        self.layer8  = basic_block(features*4,features) 
        self.bn9     = nn.BatchNorm2d(features*2)
        self.up4     = nn.ConvTranspose2d(features*2, features*2, 2, stride=2)
        
        self.layer9  = basic_block(features*4,features)
        self.out     = nn.Conv2d(features*2, 1, 1)
        
    def forward(self, x):
        
        layer1 = self.layer1(x)
        down1  = F.relu(self.down1(layer1))

        layer2 = self.layer2(down1) 
        down2  = F.relu(self.down2(layer2))
        
        layer3 = self.layer3(down2) 
        down3  = F.relu(self.down3(layer3))
        
        layer4 = self.layer4(down3) 
        down4  = F.relu(self.down4(layer4))

        
        bottleneck = self.bottleneck(down4)
        up1     = F.relu(self.up1(self.bn6(bottleneck), output_size=layer4.size()))

        merge1  = torch.cat([up1, layer4], dim=1)      
        layer6  = self.layer6(merge1)
        up2        = F.relu(self.up2(self.bn7(layer6), output_size=layer3.size()))

        merge2     = torch.cat([up2, layer3], dim=1)
        layer7     = self.layer7(merge2)
        up3        = F.relu(self.up3(self.bn8(layer7), output_size=layer2.size()))

        merge3     = torch.cat([up3, layer2], dim=1)
        layer8  = self.layer8(merge3)
        up4        = F.relu(self.up4(self.bn9(layer8), output_size=layer1.size()))
        
        merge4     = torch.cat([up4, layer1], dim=1)
        layer9  = self.layer9(merge4)                    
        out        = torch.sigmoid(self.out(layer9))
                        
        return out,layer9


import time
import math

def train_round1(model, dataloaders,augmentedLoaders, loss_fn, optimizer, acc_fn, random_state=49, epochs=1,batch_size=16,training_size=9108):
    
    
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    start = time.time()                                        #Initialize time to calculate time it takes to train model
    model.to(device)                                               #Move model to GPU     

    counter=0
    train_loss, valid_loss = [], []                            #Running training and validation loss
    val_epoch, f1_epoch = [],[]
    AP_epoch = []
    loss_list = []
    times     = []
    
    for epoch in range(epochs):
        start_epoch = time.time()
        print(f'Epoch {epoch}')
        print(scheduler.get_last_lr())
    

    #########################################Begin Model Training######################################################
    ###################################################################################################################
        
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()                             # Set training mode all the layers including batchnorm and dropout work in this
                dataloader = dataloaders['train'] 
                augmented = augmentedLoaders['train']      #get the training data
            else:
                model.eval()                              # Set model to evaluate mode deactivates the batchnorm and dropout layers
                dataloader = dataloaders['val']  
                #get the validation data

            running_loss = 0.0                            #running loss to be used for visualization later
            step = 0                                      #Batch number
            
            if phase == 'train':  
                f1 = []
                AP = []
                
                # get Range of dataloaders
                numberOfBatches = math.ceil(training_size/batch_size)
                
                for i in range(numberOfBatches):
                    
                    x1n,y1n = next(iter(dataloader))
                    x1a,y1a = next(iter(augmented))
                    
                    inputs = torch.cat((x1n,x1a),0)
                    labels = torch.cat((y1n,y1a),0)
                
#                 for inputs, labels in dataloader:
                    
                    x, y = inputs.to(device), labels.to(device)
                    step += 1

                    optimizer.zero_grad()                                   # zero the gradients
                    outputs,layer = model(x)                                      #get model output for a given input

                    #################Metrics###################
                    f1.append(acc_fn(outputs, y).cpu().detach().numpy())
                    AP.append(average_precision_score(y.reshape(-1).cpu().detach().numpy(),  outputs.reshape(-1).cpu().detach().numpy()))

                    ############################################

                    ##################Calculate Loss, backprop, and update###############
                    loss           = loss_fn(outputs, y,layer)
                    train_loss.append(loss.cpu().detach().numpy())
                    loss.backward()
                    optimizer.step()
                    print(f'Current step: {step}, AllocMem (Mb): {torch.cuda.memory_allocated()/1024/1024:.3f}, Loss: {loss:.3f},  F1: {np.mean(f1):.3f},  AP: {np.mean(AP):.3f}', end='\r') 
                    ######################################################################
        
            else:  
                loss_val = []
                f1=[]
                AP = []
                with torch.no_grad():
                    for inputs, labels in dataloader:
                        x, y = inputs.to(device), labels.to(device)
                        optimizer.zero_grad()                                   # zero the gradients
                        outputs,layer  = model(x)                                      #get model output for a given input

                        #################Metrics###################
                        f1.append(acc_fn(outputs, y).cpu().detach().numpy())
                        AP.append(average_precision_score(y.reshape(-1).cpu().detach().numpy(),  outputs.reshape(-1).cpu().detach().numpy()))

                    ############################################

                        ##################Calculate Loss, backprop, and update###############
                        valid_loss.append(loss_fn(outputs, y,layer).cpu().detach().numpy())
                        loss_val.append(valid_loss[-1])
                val_epoch.append(np.mean(loss_val))
                f1_epoch.append(np.mean(f1))
                AP_epoch.append(np.mean(AP))
                print()
                print()
                print(f' Loss val: {val_epoch[-1]:.3f}, F-Score val:{f1_epoch[-1]:.3f}, AP val:{AP_epoch[-1]:.3f} \n') 
                ######################################################################
                

            print()
            time_elapsed = time.time() - start_epoch
            times.append(time_elapsed)
            print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))    
            print('-' * 10)      

        scheduler.step()
        torch.save(model, save_last+ '\\' + f'Epoch_{str(epoch).zfill(3)}'+ '.pth')
        epoch+=1
    #########################################End Model Training######################################################
    ###################################################################################################################
    
    #Total training time including time to test
    time_elapsed = time.time() - start
    print('\n Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))    
    
#     torch.save(model, save_last+ '\\' + f'Feature32_{str(1000)}_images'+ '.pth')
    
    return {'Train Loss':train_loss,
            'Valid Loss':valid_loss,
            'Times'     :times,
            'f1_epoch':f1_epoch,
            'Epochs': epoch}
