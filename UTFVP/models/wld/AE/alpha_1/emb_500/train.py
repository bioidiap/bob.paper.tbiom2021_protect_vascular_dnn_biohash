alpha=0.1
embedding_layer_length=500
print("start the code")
import numpy as np
import random
import torch
seed=2020
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
    
all_feats = np.load('../../../../../data/all_wld_aug_feats.npy')
#all_imgs = np.load('/idiap/temp/hotroshi/paper/data/all_imgs.npy')
######### Pytorch
#all_imgs  = np.reshape(all_imgs, (all_imgs.shape[0], 1, all_imgs.shape[1], all_imgs.shape[2]))
all_feats = np.reshape(all_feats,(all_feats.shape[0], 1, all_feats.shape[1], all_feats.shape[2]))

import pickle
with open('../../../../../data/all_client_ids_aug.pkl','rb') as f:
    all_client_ids=pickle.load(f)


#all_input = all_feats[:,:,2:-2,:]/255.
#(94, 164) -> (96, 160)
all_input = np.zeros([all_feats.shape[0], all_feats.shape[1], all_feats.shape[2]+2, all_feats.shape[3]-4])
all_input[:,:,1:-1,:] = all_feats[:,:,:,2:-2]#/255.
    
print("data is ready")

from sklearn.model_selection import train_test_split
all_feats_train, all_feats_validation, all_client_ids_train, all_client_ids_validation = train_test_split(all_input, all_client_ids, test_size=0.1, random_state=42)
######################################################################

def data_gen(all_feats, all_client_ids, batch_size=32):
#     print(len(all_feats))
#     print(len(all_client_ids))
#     print(all_feats.shape)
    all_anchor = np.zeros([batch_size, 1, all_feats.shape[2], all_feats.shape[3]])
    all_pos    = np.zeros([batch_size, 1, all_feats.shape[2], all_feats.shape[3]])
    all_neg    = np.zeros([batch_size, 1, all_feats.shape[2], all_feats.shape[3]])
    
    while True:
        counter =0
        
        np.random.seed(1)
        np.random.shuffle(all_feats)
        np.random.seed(1)
        np.random.shuffle(all_client_ids)
        
        for i in range(len(all_feats)):
            rng = np.arange(len(all_feats))
            np.random.shuffle(rng)
            for k in rng:#range(i,len(all_feats)):
                if all_client_ids[i] ==  all_client_ids[k]:
                    if i != k:

                        #for j in range(len(all_feats)):
                        for j in np.random.randint(all_feats.shape[0], size=10):
                            if all_client_ids[i] !=  all_client_ids[j]:

                                #all_triplets.append([all_feats[i],all_feats[j],all_feats[k]])
                                '''
                                all_anchor.append(all_feats[i])
                                all_pos.append(all_feats[j])
                                all_neg.append(all_feats[k])
                                '''
                               # print(all_client_ids[i],all_client_ids[j],all_client_ids[k])
                                
                                all_anchor[counter,:,:,:] = all_feats[i]
                                all_pos[counter,:,:,:]    = all_feats[k]
                                all_neg[counter,:,:,:]    = all_feats[j]
                                counter +=1
                                
                                if counter ==batch_size:
                                    
                                    yield all_anchor,all_pos,all_neg
                                    
                                    all_anchor = np.zeros([batch_size, 1, all_feats.shape[2], all_feats.shape[3]])
                                    all_pos    = np.zeros([batch_size, 1, all_feats.shape[2], all_feats.shape[3]])
                                    all_neg    = np.zeros([batch_size, 1, all_feats.shape[2], all_feats.shape[3]])
                                    counter =0
                                    
                        break
                        
                                    
                                    

######################################################################
batch_size=64
train_steps_per_epoch= int(all_feats_train.shape[0]*(1)*(10)/batch_size)

import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("************ NOTE: The torch device is:", device)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1,16,3,2,padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
#             nn.Dropout(0.2),
            nn.Conv2d(16,32,3,2,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
#             nn.Dropout(0.2),
            nn.Conv2d(32,64,3,2,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
#             nn.Dropout(0.2),
            nn.Conv2d(64,128,3,2,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
#             nn.Dropout(0.2),
            nn.Conv2d(128,256,3,2,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
#             nn.Dropout(0.2),
            nn.Flatten(),
            nn.Linear(3*5*256,embedding_layer_length),
            #nn.BatchNorm2d(128),
            nn.ReLU(),
            #nn.Dropout(0.2)
        )
        
        self.fc = nn.Sequential( 
            nn.Linear(embedding_layer_length,3*5*256),
            #nn.BatchNorm2d(6*10*128),
            nn.ReLU(),
            #nn.Dropout(0.2)
        )
        
        self.decoder = nn.Sequential( 
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1), 
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), 
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), 
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), 
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1), 
            nn.Sigmoid()
        )


    def forward(self, anchor, positive, negative):
        embedded_anchor   = self.encoder(anchor)
        embedded_positive = self.encoder(positive)
        embedded_negative = self.encoder(negative)
        
        input_to_decoder_anchor   = self.fc(embedded_anchor).view(-1,256,3,5)
        input_to_decoder_positive = self.fc(embedded_anchor).view(-1,256,3,5)
        input_to_decoder_negative = self.fc(embedded_anchor).view(-1,256,3,5)
        
        decoded_anchor   = self.decoder(input_to_decoder_anchor)
        decoded_positive = self.decoder(input_to_decoder_positive)
        decoded_negative = self.decoder(input_to_decoder_negative)
        
        return decoded_anchor, decoded_positive, decoded_negative, embedded_anchor, embedded_positive, embedded_negative
        
    def get_embeding(self,x):
        return self.encoder(x)
        
model = Net()
model.to(device)
model.train()
if device =='cuda':
    model = model.cuda()
    
triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)

def AE_loss(decoded, native):
    MAE = nn.L1Loss()
    MSE = nn.MSELoss()
    BCE = nn.BCELoss()
    return BCE(decoded, native)# 0.5*MAE(decoded, native)+ MSE(decoded, native)

learning_rate=1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                             weight_decay=1e-5)

lr_e={0:1e-3,
      1:1e-3,
      2:5e-4,
      3:5e-4,
      4:1e-4,
      5:1e-4,
      6:5e-5,
      7:5e-5,
      8:1e-5,
      9:1e-5}
#optimizer.param_groups[0]["lr"]=lr_e[0]#0.0001


import os
if not os.path.exists('model'):
    os.makedirs('model')
#########################################################################


def evaluate_model(all_feats, all_client_ids, device):
    
    np.random.seed(1)
    np.random.shuffle(all_feats)
    np.random.seed(1)
    np.random.shuffle(all_client_ids)

    counter =0
    loss_values = np.zeros(15)
    for i in range(len(all_feats)):
        rng = np.arange(len(all_feats))
        np.random.shuffle(rng)
        for k in rng:#range(i,len(all_feats)):
            if all_client_ids[i] ==  all_client_ids[k]:
                if i != k:

                    #for j in range(len(all_feats)):
                    for j in np.random.randint(all_feats.shape[0], size=10):
                        
                        if all_client_ids[i] !=  all_client_ids[j]:    
                            anchor   = np.reshape(all_feats[i], (1,1,all_feats.shape[2],all_feats.shape[3]))
                            positive = np.reshape(all_feats[k], (1,1,all_feats.shape[2],all_feats.shape[3]))
                            negative = np.reshape(all_feats[j], (1,1,all_feats.shape[2],all_feats.shape[3]))
                            
                            anchor   = torch.tensor(anchor, requires_grad=False).float().to(device)
                            positive = torch.tensor(positive, requires_grad=False).float().to(device)
                            negative = torch.tensor(negative, requires_grad=False).float().to(device)
                            
                            counter +=1
                            #print(i,counter,all_client_ids[i],all_client_ids[j],all_client_ids[k])
                            
                            decoded_anchor, decoded_positive, decoded_negative, embedded_anchor, embedded_positive, embedded_negative = model.forward(anchor, positive, negative)
                   
                            distance_anchor_positive   = (embedded_anchor - embedded_positive).pow(2).mean()  # .pow(.5)
                            distance_anchor_negative   = (embedded_anchor - embedded_negative).pow(2).mean()  # .pow(.5)
                            distance_positive_negative = (embedded_positive - embedded_negative).pow(2).mean()
                            
                            embeding_triplet_loss = triplet_loss(embedded_anchor, embedded_positive, embedded_negative)

                            anchor_AE_loss   = AE_loss(decoded_anchor,   anchor  )
                            positive_AE_loss = AE_loss(decoded_positive, positive)
                            negative_AE_loss = AE_loss(decoded_negative, negative)

                            anchor_MAE   = nn.L1Loss()(decoded_anchor,   anchor  )
                            positive_MAE = nn.L1Loss()(decoded_positive, positive)
                            negative_MAE = nn.L1Loss()(decoded_negative, negative)

                            anchor_MSE   = nn.MSELoss()(decoded_anchor,   anchor  )
                            positive_MSE = nn.MSELoss()(decoded_positive, positive)
                            negative_MSE = nn.MSELoss()(decoded_negative, negative)

                            total_loss =  embeding_triplet_loss + 0.2*(anchor_AE_loss + positive_AE_loss + negative_AE_loss)
                            
                            total_AE_loss = anchor_AE_loss + positive_AE_loss + negative_AE_loss
                            loss_values += np.array([total_loss.data.item(), total_AE_loss.data.item(),\
                                                     anchor_AE_loss.item(), positive_AE_loss.item(), negative_AE_loss.item(), 
                                                     anchor_MAE.item(), positive_MAE.item(), negative_MAE.item(),\
                                                     anchor_MSE.item(), positive_MSE.item(), negative_MSE.item(),\
                                                     embeding_triplet_loss.item(),\
                                                     distance_positive_negative.item(), distance_anchor_positive.item(),\
                                                     distance_anchor_negative.item()])
                    break
                            
                            
    return loss_values/float(counter)

################################################################################
with open('models_log.csv','w') as f:
    f.write('epoch, lr, total loss(train), total_AE_loss(train), AE_loss_anchor(train), AE_loss_positive(train), AE_loss_negative(train), anchor_MAE(train), pos_MAE(train), neg_MAE(train), anchor_MSE(train), pos_MSE(train), neg_MSE(train), triplet(train), p/n(train), a/p(train), a/n(train), total loss(validation), total_AE_loss(validation), AE_loss_anchor(validation), AE_loss_positive(validation), AE_loss_negative(validation), anchor_MAE(validation), pos_MAE(validation), neg_MAE(validation), anchor_MSE(validation), pos_MSE(validation), neg_MSE(validation), triplet(validation), p/n(validation), a/p(validation), a/n(validation) \n')


num_epochs=100
all_itrations = 0
for epoch in range(num_epochs):
    optimizer.param_groups[0]["lr"]=lr_e[int(epoch/10)]
    iteration=0
    for data in data_gen(all_feats_train, all_client_ids_train, batch_size):
        anchor, positive, negative = data
        anchor   = torch.tensor(anchor, requires_grad=False).float().to(device)
        positive = torch.tensor(positive, requires_grad=False).float().to(device)
        negative = torch.tensor(negative, requires_grad=False).float().to(device)
        
        # ===================forward=====================
        optimizer.zero_grad()
        decoded_anchor, decoded_positive, decoded_negative, embedded_anchor, embedded_positive, embedded_negative = model.forward(anchor, positive, negative)
        
                
        embeding_triplet_loss = triplet_loss(embedded_anchor, embedded_positive, embedded_negative)
        
        anchor_AE_loss   = AE_loss(decoded_anchor,   anchor  )
        positive_AE_loss = AE_loss(decoded_positive, positive)
        negative_AE_loss = AE_loss(decoded_negative, negative)
        total_AE_loss = anchor_AE_loss + positive_AE_loss + negative_AE_loss

        
        anchor_MAE   = nn.L1Loss()(decoded_anchor,   anchor  )
        positive_MAE = nn.L1Loss()(decoded_positive, positive)
        negative_MAE = nn.L1Loss()(decoded_negative, negative)
        
        anchor_MSE   = nn.MSELoss()(decoded_anchor,   anchor  )
        positive_MSE = nn.MSELoss()(decoded_positive, positive)
        negative_MSE = nn.MSELoss()(decoded_negative, negative)
        
        total_loss =  alpha*embeding_triplet_loss + (1-alpha)*(total_AE_loss)
        # ===================backward====================
        #optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        #print(total_loss.data.item(), embeding_triplet_loss.item())
        
        
        # ===================log========================
        iteration +=1
        all_itrations+=1
                
        if iteration==train_steps_per_epoch:
            break
            
            
    torch.save(model.state_dict(), 'model/{}.pth'.format(epoch+1))
    
    evaluate_train = evaluate_model(all_feats_train, all_client_ids_train, device)
    evaluate_validation = evaluate_model(all_feats_validation, all_client_ids_validation, device)
    with open('models_log.csv','a') as f:
        f.write('{}, {:.5f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}\n'.format(epoch+1,  optimizer.param_groups[0]["lr"], \
                evaluate_train[0], evaluate_train[1], evaluate_train[2],\
                evaluate_train[3], evaluate_train[4], evaluate_train[5],\
                evaluate_train[6], evaluate_train[7], evaluate_train[8],\
                evaluate_train[9], evaluate_train[10], evaluate_train[11],\
                evaluate_train[12], evaluate_train[13], evaluate_train[14],\
                evaluate_validation[0], evaluate_validation[1], evaluate_validation[2],\
                evaluate_validation[3], evaluate_validation[4], evaluate_validation[5],\
                evaluate_validation[6], evaluate_validation[7], evaluate_validation[8],\
                evaluate_validation[9], evaluate_validation[10], evaluate_validation[11],\
                evaluate_validation[12], evaluate_validation[13], evaluate_validation[14]))