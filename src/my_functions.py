# pre-trained models
from tensorflow.keras.applications import VGG16, ResNet50, VGG19
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Conv2D, Dense, Dropout, MaxPooling2D, Flatten #, ZeroPadding2D, BatchNormalization, M
from keras.layers import GlobalAveragePooling2D # GlobalMaxPooling2D,
from tensorflow.keras.optimizers import SGD

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


from tensorflow.keras import layers
#from tensorflow.keras import Model

def freezing_layers(model, block: str):
  '''
  for transfer-learning
  it freezes all layers until the selected block
  '''
  for layer in model.layers:
    if layer.name == block:
      break

    layer.trainable = False
    print("layer " + layer.name + " frozen")
    
def nn_parameters():
    '''
    To assign nn parameters 
    '''
    #Initializing the hyperparameters
    batch_size= 128
    epochs=200
    learn_rate=.01
    sgd=SGD(learning_rate=learn_rate,momentum=.9,nesterov=False)
    loss_type='categorical_crossentropy'
    #adam=Adam(learning_rate=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    return batch_size, epochs, learn_rate, sgd, loss_type

def vgg16_19_conf(name: str, neurons_final_layer:int, dropout_layers: bool, dropout_percent: float, y_train):
  '''
  creates a vgg16/19 architecture ready for transfer-learning (all layers frozen but top model). Choose the number of neurons in the last layer, if including
  dropout layers after dense layers and percentaje
  options: vgg16, vgg19
  EX: vgg16_19_conf("vgg16",5, True, 0.2)
  '''

  if name =="vgg16":
    base_model = VGG16(include_top=False, weights='imagenet', input_shape=(128, 128, 3), classes = y_train.shape[1], classifier_activation='softmax')
  elif name == "vgg19":
    base_model = VGG19(include_top=False, weights='imagenet', input_shape=(128, 128, 3), classes = y_train.shape[1], classifier_activation='softmax')
  else:
    raise ValueError("check descripcion and choose a proper one")

  model = Sequential() 
  # adding base model layers to our architecture
  for layer in base_model.layers:
    model.add(layer) 
    layer.trainable=False

  # for layer in model.layers:
  #   layer.trainable=False

  # top model
  model.add(Flatten())
  model.add(Dense(4096,activation=('relu')))
  if dropout_layers==True:
    model.add(Dropout(0.2)) 
    model.add(Dense(4096,activation=('relu'))) 
    model.add(Dropout(0.2)) 
  else:
    model.add(Dense(4096,activation=('relu'))) 
  model.add(Dense(neurons_final_layer,activation=('softmax'))) 
  return model
  
def generic_last_2layers(nn,neurons_final_layer:int, dropout_layers: bool, dropout_percent: float, y_train):
  pre_trained = Sequential()
  pre_trained.add(nn)

  # Freeze the layers 
  for layer in pre_trained.layers:
      layer.trainable = False

  #aÃ±adimos el top model
  pre_trained.add(GlobalAveragePooling2D()) # https://keras.io/api/layers/pooling_layers/global_average_pooling2d/
  if dropout_layers ==True:
    pre_trained.add(Dropout(0.2)) 
  pre_trained.add(Dense(5,activation=('softmax')))  # sustituyo  por mi Ãºltima capa de 5neuronas => pre_trained_resnet50.add(Dense(1024,activation=('softmax'))) 
  return pre_trained
  
def resnet50_conf(neurons_final_layer:int, dropout_layers: bool, dropout_percent: float, y_train):
  pre_trained_resnet50 = Sequential()
  pre_trained_resnet50.add(ResNet50(include_top=False, weights='imagenet', input_shape=(128, 128, 3), classes = y_train.shape[1], classifier_activation='softmax'))

  # Freeze the layers 
  for layer in pre_trained_resnet50.layers:
      layer.trainable = False

  #aÃ±adimos el top model
  pre_trained_resnet50.add(GlobalAveragePooling2D()) # https://keras.io/api/layers/pooling_layers/global_average_pooling2d/
  if dropout_layers ==True:
    pre_trained_resnet50.add(Dropout(0.2)) 
  pre_trained_resnet50.add(Dense(5,activation=('softmax')))  # sustituyo  por mi Ãºltima capa de 5neuronas => pre_trained_resnet50.add(Dense(1024,activation=('softmax'))) 
  return pre_trained_resnet50
  
  
def plotting_model(model, epochs, name):
  #Plotting the training and validation loss and accuracy
  f,ax=plt.subplots(2,1) 

  #Loss
  ax[0].plot(np.arange(0, epochs), model.history["loss"], label="train_loss")
  ax[0].plot(np.arange(0, epochs), model.history["val_loss"], label="val_loss")
  ax[0].legend()
  plt.xlabel("Epoch #")
  plt.ylabel("Loss/Accuracy")
  #Accuracy
  ax[1].plot(np.arange(0, epochs), model.history["accuracy"], label="train_acc")
  ax[1].plot(np.arange(0, epochs), model.history["val_accuracy"], label="val_acc")
  ax[1].legend()
  #plt.legend()
  #plt.title("Training Loss and Accuracy")
  plt.xlabel("Epoch ||    "+name)
  plt.ylabel("Loss/Accuracy")
  plt.show()

