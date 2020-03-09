#!/usr/bin/env python
## \file
## \ingroup tutorial_tmva_keras
## \notebook -nodraw
## This tutorial shows how to do CNN classification in TMVA with neural networks
## trained with keras.
##
## \macro_code
##
## \date 2019
## \author TMVA Team

from ROOT import TMVA, TFile, TString, TTree, TCut
from subprocess import call
from os.path import isfile

## to use tensorflow backend
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import time
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import backend as K

from keras.regularizers import l2
from keras.optimizers import SGD

TMVA.Tools.Instance()
TMVA.PyMethodBase.PyInitialize()

## For PYMVA methods
TMVA.PyMethodBase.PyInitialize()


outputFile = TFile.Open("CNN_ClassificationOutput.root", "RECREATE")

factory = TMVA.Factory("TMVA_CNN_Classification", outputFile,
                      "!V:ROC:Silent:Color:!DrawProgressBar:AnalysisType=Classification" )

# Load data
if not isfile('sample_images_32x32.root'):
    call(['curl', '-o', 'sample_images_32x32.root', 'https://cernbox.cern.ch/index.php/s/mba2sFJ3ugoy269/download'])

data = TFile.Open('sample_images_32x32.root')
signal = data.Get('sig_tree')
background = data.Get('bkg_tree')

dataloader = TMVA.DataLoader('dataset')
imgsize = 32 * 32
for i in range(imgsize):
    varName = "var_{} := vars[{}]".format(i,i)
    dataloader.AddVariable(varName,'F')

dataloader.AddSignalTree(signal, 1.0)
dataloader.AddBackgroundTree(background, 1.0)
trainTestSplit = 0.8
dataloader.PrepareTrainingAndTestTree( TCut(''),
                                  "nTrain_Signal=8000:nTrain_Background=8000:SplitMode=Random:"
                                   "NormMode=NumEvents:!V" )

################ KERAS MODEL ###############################
# Generate model
batch_size = 64
num_classes = 2
epochs = 15

# Define model
model = Sequential()
model.add(Reshape((32, 32, 1), input_shape=(imgsize,)))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Set loss and optimizer
model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=0.0003, decay=1e-6, momentum=0.9),
              metrics=['accuracy'])

# Store model to file
model.save('model_cnn.h5')
model.summary()

# Book methods
factory.BookMethod(dataloader, TMVA.Types.kPyKeras, 'PyKeras',
                   'H:!V:VarTransform=None:FilenameModel=model_cnn.h5:'
                   'FileNameTrainedModel=trained_model_cnn.h5:NumEpochs={}:BatchSize={}'.format(epochs, batch_size))

############### TMVA model #######################3#input layout 
inputLayoutString = "InputLayout=1|32|32"
                                                                                                
## Batch Layout                                                                                                                                     
batchLayoutString = "BatchLayout=64|1|1024"
                                                   
layoutString = ("Layout=CONV|8|3|3|1|1|1|1|RELU,BNORM,CONV|8|3|3|1|1|1|1|RELU,"
                "BNORM,RESHAPE|FLAT,DENSE|64|TANH,DENSE|1|TANH")

##Training strategies.                                                                                                                          
training1 = ("LearningRate=1e-4,Momentum=0.9,Repetitions=1,"
                     "ConvergenceSteps=10,BatchSize=64,TestRepetitions=1,"
                     "MaxEpochs=20,WeightDecay=1e-4,Regularization=None,"
                     "Optimizer=SGD,DropConfig=0.0+0.0+0.0+0.0")
 
trainingStrategyString = "TrainingStrategy=" + training1
    
## General Options.                                                                                                                              
cnnOptions = ("!H:V:ErrorStrategy=CROSSENTROPY:VarTransform=None:"
                       "WeightInitialization=XAVIERUNIFORM")

cnnOptions +=  ":" + inputLayoutString
cnnOptions +=  ":" + batchLayoutString
cnnOptions +=  ":" + layoutString
cnnOptions +=  ":" + trainingStrategyString
cnnOptions +=  ":Architecture=CPU"

##book CNN
factory.BookMethod(dataloader, TMVA.Types.kDL, "DL_CNN", cnnOptions)


# Run training, test and evaluation
factory.TrainAllMethods()
factory.TestAllMethods()
factory.EvaluateAllMethods()

# Print ROC
canvas = factory.GetROCCurve(dataloader)
canvas.Draw()

time.sleep(300)
