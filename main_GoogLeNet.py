
# TO DO

# -- Download the data 
# -- Extract the file
# -- Split the File

# Pytorch Modules
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
import torchvision.datasets as datasets
import torchvision.transforms as transforms 
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

#General Modules
from PIL import Image
from tqdm import tqdm
from time import time, sleep
import numpy as np 
import os
import sys
from glob import glob
import pandas as pd
import shutil
# Importing Model

from GoogLeNet import GoogLeNet



START_SESSION = time()

all_images = glob('Dataset/*/*/*/*.png')

all_images = glob('Dataset_custom_build/*/*.png')
emotions = {'anger': 0, 'disgust':1,'fear':2, 'joy':3, 'neutral':4, 'sadness':5, 'surprise':6}

class FERG_2D_256_x_256_Dataset(Dataset):
    def __init__(self):
        self.imgdata = []
        self.labels = []
        self.arr = None 

        for emotion_key in tqdm(emotions.keys(), desc = "Dataset creations", leave = False):
            
            emotion_val = emotions[emotion_key]
            emotion_key_filtered = list(filter(lambda x: emotion_key in x, all_images))
            
            for img in tqdm(emotion_key_filtered, leave = False):
                try:
                    self.arr = Image.open(img).convert('RGB')
                    self.arr = self.arr.resize((224, 224))
                    self.arr = np.asarray(self.arr)
                    self.arr = self.arr.reshape((3,224,224))
                    self.imgdata.append(self.arr)
                    self.labels.append(emotion_val)
                except:
                    pass
        self.imgdata = np.array(self.imgdata)

        self.X =torch.tensor(self.imgdata,  dtype = torch.float32)
        self.Y =torch.tensor(np.array(self.labels),dtype = torch.long)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()  
        return self.X[idx], self.Y[idx] 



def save_checkpoint(state, filename):
    print("Saving model checkpoint")
    torch.save(state, filename)


def load_checkpoint(model, optimizer, checkpoint):
    print("Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])



def accuracy_check(predictions, labels):
    classes = torch.argmax(predictions, dim=1)
    # return torch.mean((classes == labels).float())
    return (classes == labels).float().mean().item()

input_size    = 1*224*224
num_classes   = 7
batch_size    = 8
learning_rate = 3e-4
num_epochs    = 1
checkpoint_file = "my_checkpoint.pth.tar"
load_model    = True if os.path.isfile("my_checkpoint.pth.tar") else False



print("Dataset Creation")
train_dataset = FERG_2D_256_x_256_Dataset()

print("Dataset Creation succesfull")

print("Dataset loading")
train_loader  = DataLoader(train_dataset    , batch_size = batch_size,  shuffle = True)
print("Dataset Loading succesfull")

# check if cuda is available
device = 'cuda'if torch.cuda.is_available() else 'cpu'


model = GoogLeNet(in_channels = 3, num_classes = num_classes).to(device)

# Loss Function and Optimizers
criterion  = nn.CrossEntropyLoss()
optimizer  = optim.Adam(model.parameters(), lr = learning_rate)


writer = SummaryWriter(f"runs/EmotionDetectionFERGB_2D_256_x_256/GoogLeNet") 

def train():
    losses = []
    accuracies = []
    step = 0
    for epoch in range(num_epochs):                # Initializing or setting to the training method
        model.train()
        loop = tqdm(enumerate(train_loader), total = len(train_loader), leave = False)
        for batch_idx, (data, target) in loop:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            accuracy = accuracy_check(output,target)
            loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
            loop.set_postfix(loss = loss.item(), accu = "{0:.0%}".format(accuracy))            
            writer.add_scalar('Trainig loss', loss.item(), global_step = step)
            writer.add_scalar('Trainig Acc.', accuracy, global_step = step)

            step +=1
            sleep(0.002)





# def test():
#     model.eval()
#     test_loss = 0
#     correct   = 0
#     with torch.no_grad():
#         loop = tqdm(enumerate(val_loader), total = len(val_loader), leave = False)
#         for batch_idx, (data, target) in loop:
#             print("Evaluating...", end = '\r' )
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             loss = criterion(output, target)
#             test_loss += loss.item()
#             pred = output.argmax(dim=1, keepdim=True)
#             correct += pred.eq(target.view_as(pred)).sum().item()
#             # losses.append(loss.item())
#             accuracy = accuracy_check(output,target)
#             loop.set_description("Validation set")
#             loop.set_postfix(loss = loss.item(), accu = "{0:.0%}".format(accuracy))

#     test_loss /= len(val_loader.dataset) 
#     print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         test_loss, correct, len(val_loader.dataset),
#         100. * correct / len(val_loader.dataset)))


def main():
    start = time()
    train()
    print('\n Time Taken {:.2f} secs'. format(time()-start))
    start = time()
    print('\n Time Taken {:.2f} secs'. format(time()-start))
    # test()

if __name__ == '__main__':
    main()
    print(f"TOOK {time()-START_SESSION} SECS TO RUN ENTIRE MODULE !!!")
    torch.save(model.state_dict(), "EmotionDetection_FERG_DB_256_GoogLeNet_model.pt")
