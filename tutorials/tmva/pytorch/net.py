"""
Module implementing the CNN to classify between the two types of images
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import json
import numpy as np
import datetime
from tqdm import tqdm

class ConvBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=3, batchnorm=True, padding=1):
        super(ConvBlock, self).__init__()
        self.c1 = nn.Conv2d(in_filters, out_filters, kernel_size, padding=padding)
        self.bn = batchnorm
        self.b1 = nn.BatchNorm2d(out_filters)
        
    def forward(self, x):
        x = self.c1(x)
        if self.bn: 
            x = self.b1(x)
        return x
    
class Net(nn.Module):
    def __init__ (self, imgsize=32, num_classes=10, in_filters=1, filters=8, kernel_size=3, batchnorm=True, activ=F.relu, padding=1):
        super(Net, self).__init__()

        self.activ = activ
        self.bn = batchnorm
        self.imgsize = imgsize
        
        # Convolutional layers
        self.activ = activ
        self.conv1 = ConvBlock(in_filters=in_filters, out_filters=filters, 
                               kernel_size=kernel_size, batchnorm=batchnorm, padding=padding)
        self.conv2 = ConvBlock(in_filters=filters, out_filters=filters, 
                               kernel_size=kernel_size, batchnorm=batchnorm, padding=padding)
        
        self.fc1 = nn.Linear(filters * imgsize ** 2, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.activ(self.conv1(x))
        x = self.activ(self.conv2(x))
        
        x = x.view(-1, self.num_flat_features(x))
        x = self.activ(self.fc1(x))
        x = self.fc2(x)
        return x
        
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class Trainer:
    def __init__ (self, imgsize=32, num_classes=10, in_filters=1, filters=8, kernel_size=3, batchnorm=True, activ=F.relu):
        """
        Trainer for the CNN

        @num_epochs: Number of epochs
        @lr: initial learning rate
        @batch_size: Size of single batch
        @decay: decay LR by 3 after these many epochs
        """
        self.imgsize = imgsize
        self.net = Net(imgsize=imgsize, num_classes=num_classes, in_filters=in_filters, filters=filters, kernel_size=kernel_size,
                       batchnorm=batchnorm, activ=activ)
        
        self.cuda_flag = torch.cuda.is_available()
        if self.cuda_flag:
            self.net = self.net.cuda()

    def train(self, X, Y, epochs=10, lr=0.001, batch_size=64, decay=5000, logging=True, log_file=None):
        """
        Train the CNN

        Params:
        @X: Training data - input of the model
        @Y: Training labels
        @logging: True for printing the training progress after each epoch
        @log_file: Path of log file
        """
        X = [x.reshape(1, self.imgsize, self.imgsize) for x in X]
        
        inputs = torch.FloatTensor(X)
        labels = torch.LongTensor(Y)
        
        train_dataset = TensorDataset(inputs, labels)
        trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        self.net.train()
        criterion = nn.CrossEntropyLoss()

        for epoch in range(1, epochs+1): # loop over data multiple times
            # Decreasing the learning rate
            if (epoch % decay == 0):
                lr /= 3
                
            optimizer = optim.SGD(self.net.parameters(), lr=lr, momentum=0.9)
            
            tot_loss = 0.0
            for data in tqdm(trainloader):
                # get the inputs
                inputs, labels = data
                if self.cuda_flag:
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                o = self.net(inputs)
                loss = criterion(o, labels)

                loss.backward()
                optimizer.step()
                
                tot_loss += loss.item()
                
            tot_loss /= len(trainloader)

            # logging statistics
            timestamp = str(datetime.datetime.now()).split('.')[0]
            log = json.dumps({
                'timestamp': timestamp,
                'epoch': epoch,
                'loss': float('%.7f' % tot_loss),
                'lr': float('%.6f' % lr)
            })
            if logging:
                print (log)

            if log_file is not None:
                with open(log_file, 'a') as f:
                    f.write("{}\n".format(log))
            
        print ('Finished Training')

    def predict(self, inputs):
        """
        Predict the task labels corresponding to the input images
        """
        inputs = [x.reshape(1, self.imgsize, self.imgsize) for x in inputs]
        inputs = torch.FloatTensor(inputs)
        if self.cuda_flag:
            inputs = inputs.cuda()

        self.net.eval()
        with torch.no_grad():
            labels = self.net(inputs).cpu().numpy()
            
        return np.argmax(labels, axis=1)

    def score(self, X, Y):
        """
        Score the model -- compute accuracy
        """
        pred = self.predict(X)
        acc = np.sum(pred == Y) / len(Y)
        return float(acc)

    def save_model(self, checkpoint_path, model=None):
        if model is None: model = self.net
        torch.save(model.state_dict(), checkpoint_path)
    
    def load_model(self, checkpoint_path, model=None):
        if model is None: model = self.net
        if self.cuda_flag:
            model.load_state_dict(torch.load(checkpoint_path))
        else:
            model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))

