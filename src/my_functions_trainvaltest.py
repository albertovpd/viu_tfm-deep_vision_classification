#%tensorflow_version 2.ximage_dataset_from_directory

# nns
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
#from sklearn.metrics import plot_precision_recall_curve
import json # to save in a file metrics
#from datetime import datetime # to name results

# viz & arrays
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline

# navigating through folder
import os

def vgg19_vgg16(data_augmentation, base_model, dropout_layers: bool, dropout_position: str, dropout_percent: float, num_classes):
  '''
  both architectures has the same top model, so we'll built:
  - a model with/without a data augmentation layer
  - the functional nn with frozen layers, whithout top model
  - "empty" top model layers, ready for transfer-learning, with/without dropout in first/middle layer of top model
  - final layer with as much neurons as our classes to infer
  EX: vgg19(data_augmentation=None, base_model= base_model_vgg19, dropout_layers=False, dropout_position= None, dropout_percent=None, num_classes=5)
  '''
  pre_trained = Sequential()

  if data_augmentation:
    #data augmentation
    pre_trained.add(data_augmentation)
    pre_trained.add(layers.Rescaling(1./255))
       
                                                                 
  # vgg16 (Functional)          
  pre_trained.add(base_model)

  # Freeze the layers 
  for layer in pre_trained.layers:
      layer.trainable = False

  # i had to insert this layer when using the data augmentation layer in order to avoid dimension errors with VGG16 and 19
  pre_trained.add(layers.GlobalAveragePooling2D())

  # adding top model with/without dropout in first/middle position
  # top moddel for vgg19-16 are a flatten layer, 2 dense layers of 4096n and pred layer 
  pre_trained.add(layers.Flatten())
  if dropout_layers is True:
      if dropout_position=="first":
          pre_trained.add(layers.Dropout(dropout_percent)) 
          pre_trained.add(layers.Dense(4096,activation=('relu')))

      elif dropout_position == "middle":                  
          pre_trained.add(layers.Dense(4096,activation=('relu')))
          pre_trained.add(layers.Dropout(dropout_percent)) 
      
  else: 
    pre_trained.add(layers.Dense(4096,activation=('relu')))
  
  pre_trained.add(layers.Dense(4096,activation=('relu')))
  pre_trained.add(layers.Dense(num_classes,activation=('softmax')))

  return pre_trained

def generic_last_2layers(data_augmentation, nn,neurons_final_layer:int,  dropout_layers: bool, dropout_position: str =="first", dropout_percent: float):
  '''
  Xception, InceptionResNetV2, DenseNet121 have different architecture, but all of them have the same top model. So we add the data augmentation layer,
  load the functional model and add the top model with/without dropout in first/middle position
  works with pre-trained models (Xception, InceptionResNetV2, DenseNet121) for transfer-learning. 
  top layer consisting of 2 layers: globalaverage2d layer and predictions layer.
  - nn                  = pre-trained model without top model
  - neurons_final_layer = how many classes we want to work with
  - dropout_layers      = bool. dropout in the top model (True/False)
  - dropout_position    = dropout layer before top model or between dense layers (first/middle)
  - dropout_percent     = (0,1). float
  EX: generic_last_2layers(Xception(include_top=False, 
                                    weights='imagenet', 
                                    input_shape=(128, 128, 3), 
                                    classes = num_classes, 
                                    classifier_activation='softmax'
                                    ), 
                          5, True, "first", 0.2
                          )
  '''
  pre_trained = Sequential()

  if data_augmentation:
    #data augmentation
    pre_trained.add(data_augmentation)
    pre_trained.add(layers.Rescaling(1./255))

  pre_trained.add(nn)

  # Freeze the layers 
  for layer in pre_trained.layers:
      layer.trainable = False

  #adding top model with/without dropout
  if dropout_layers is True:
      if dropout_position=="first":
          pre_trained.add(layers.Dropout(dropout_percent)) 
          pre_trained.add(layers.GlobalAveragePooling2D()) # https://keras.io/api/layers/pooling_layers/global_average_pooling2d/
          

      elif dropout_position == "middle":
          pre_trained.add(layers.GlobalAveragePooling2D()) 
          pre_trained.add(layers.Dropout(dropout_percent)) 
      
  else:
    pre_trained.add(layers.GlobalAveragePooling2D())
  
  # last layer
  pre_trained.add(layers.Dense(neurons_final_layer,activation=('softmax')))
  return pre_trained


def plotting_model(model, epochs, name, location):
  #Plotting the training and validation loss and accuracy
  
  f,ax=plt.subplots(2, 1, figsize=(15,10))
  #Loss
  ax[0].plot(np.arange(0, epochs), model.history["loss"], label="train_loss")
  ax[0].plot(np.arange(0, epochs), model.history["val_loss"], label="val_loss")
  ax[0].grid(visible=True, which="both", axis='both')
  ax[0].legend()

  plt.xlabel("Epoch #")
  plt.ylabel("Loss/Accuracy")
  #Accuracy
  ax[1].plot(np.arange(0, epochs), model.history["accuracy"], label="train_acc")
  ax[1].plot(np.arange(0, epochs), model.history["val_accuracy"], label="val_acc")
  ax[1].grid(visible=True, which="both", axis='both')
  ax[1].legend()

  plt.xlabel("Run Epochs: "+str(epochs)+" ||    "+name)
  plt.ylabel("Loss/Accuracy")
  
  plt.savefig(location)
  plt.show()

def model_evaluation(evaluation, output_folder:str, name:str):
  # evaluation the model with val_ds
  models_metrics = {}  
  models_metrics[m] = evaluation
  # saving the metris in json file
  with open(output_folder+name+"-metrics.json", "w") as outfile:
    json.dump(models_metrics, outfile)
  return models_metrics

def classification_report_pic(y_pred, validation_ds, class_names, output_folder, name):
  numeric_values_val = list(validation_ds.map(lambda x, y: y))
  y_target = []
  for arr in numeric_values_val:
    y_target.append(arr)
  y_target = list(chain.from_iterable(y_target))
  # transform to np array
  y_target = np.array(y_target)
  #print(y_target.shape)

  print(classification_report(y_pred , y_target, target_names=class_names))
  clf_report = classification_report(y_pred , y_target, target_names=class_names, output_dict=True )
  # .iloc[:-1, :] to exclude support
  plt.figure(figsize = (10,8))
  plt.title("classification report: "+name)
  sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True)
  plt.savefig(output_folder+name+"-classification_report.png")
  plt.show()

def confusion_matrix_report(y_pred, validation_ds,  class_names, output_folder, name):
    #cm = confusion_matrix(y_pred, y_target)
    true_categories = tf.concat([y for x, y in val_ds], axis=0)
    cm = confusion_matrix(true_categories, y_pred, normalize="true" ) #,  labels=[class_names])
    df_cm = pd.DataFrame(cm, index= [class_names], columns = [class_names])
    fig = plt.figure(figsize = (10,8))
    ax1 = fig.add_subplot(1,1,1)
    sns.set(font_scale=1.4) #for label size
    sns.heatmap(df_cm, annot=True, annot_kws={"size": 12})
    plt.title("confusion matrix: "+name)
    ax1.set_ylabel('True Values',fontsize=14)
    ax1.set_xlabel('Predicted Values',fontsize=14)
    plt.savefig(output_folder+name+"-confusion_matrix.png")
    plt.show()