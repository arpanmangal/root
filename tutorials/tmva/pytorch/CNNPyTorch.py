#!/usr/bin/env python
## \file
## \ingroup tutorial_tmva_keras
## \notebook -nodraw
## This example shows how to do classification with CNN
## trained with PyTorch.
##
## \macro_code
##
## \date 2019
## \author TMVA Team

from subprocess import call
from os.path import isfile

from reader import get_dataset
from net import Trainer

# Load data
if not isfile('sample_images_32x32.root'):
    call(['curl', '-o', 'sample_images_32x32.root', 'https://cernbox.cern.ch/index.php/s/mba2sFJ3ugoy269/download'])

(X_train, Y_train), (X_test, Y_test) = get_dataset('sample_images_32x32.root', test_split=0.2)
print ("Loaded data")

# Training configs
num_epochs = 60
lr = 0.0001
batch_size = 64

trainer = Trainer(imgsize=32, num_classes=2)
trainer.train(X_train, Y_train, lr=lr, epochs=num_epochs, batch_size=batch_size)
trainer.save_model('model_cnn.pth')

print ('Evaluating...')
print ('Training Acc.: {:.4f} %'.format(trainer.score(X_train, Y_train) * 100))
print ('Test Acc.    : {:.4f} %'.format(trainer.score(X_test, Y_test) * 100))
