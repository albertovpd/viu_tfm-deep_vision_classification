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


def plotting_model(fitted_model, epochs, name, location, testval_label="test"):
  '''
  Plotting the training and validation loss and accuracy
  :model: nn
  :epochs
  :name: name to save the final image
  :location: folder where the picture will be stored to save the image
  :testval_label: for displaying "test" or "val" label while using the test/label set
  ex: plotting_model(history,number_of_epochs_it_ran, name, output_folder, "test") 
  '''

  f,ax=plt.subplots(2, 1, figsize=(15,10))
  #Loss
  ax[0].plot(np.arange(0, epochs), fitted_model.history["loss"], label="train_loss")
  
  if testval_label == "test":
    ax[0].plot(np.arange(0, epochs), fitted_model.history["val_loss"], label="test_loss")
    ax[0].grid(visible=True, which="both", axis='both')
    ax[0].legend()
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    #Accuracy
    ax[1].plot(np.arange(0, epochs), fitted_model.history["accuracy"], label="train_acc")
    ax[1].plot(np.arange(0, epochs), fitted_model.history["val_accuracy"], label="test_acc")

  elif testval_label == "val":
    ax[0].plot(np.arange(0, epochs), fitted_model.history["val_loss"], label="val_loss")
    ax[0].grid(visible=True, which="both", axis='both')
    ax[0].legend()
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    #Accuracy
    ax[1].plot(np.arange(0, epochs), fitted_model.history["accuracy"], label="train_acc")
    ax[1].plot(np.arange(0, epochs), fitted_model.history["val_accuracy"], label="val_acc")
  
  ax[1].grid(visible=True, which="both", axis='both')
  ax[1].legend()
  
  plt.xlabel("Run Epochs: "+str(epochs)+"| "+name)
  plt.ylabel("Loss/Accuracy")  
    
  save_path = location+name+"-loss_accuracy.png"
  plt.savefig(save_path)
  plt.show()
  

def model_evaluation(evaluation, output_folder:str, name:str):
  '''
  evaluation the model with test_ds
  :evaluation: model.evaluate(val_dataset, batch_size, return_dict=True)
  :output_folder: path to save the json
  :name: name of the file
  '''
  models_metrics = {}  
  models_metrics[name] = evaluation
  # saving the metris in json file
  with open(output_folder+name+"-metrics.json", "w") as outfile:
    json.dump(models_metrics, outfile)
  return models_metrics

def classification_report_pic(y_pred, y_target, class_names, output_folder, name):
  '''
  prints inline the classification report and also saves a pic with results
  - y_pred = inferences: n.argmax( model.predict(test_ds), axis=1)
  - validation_ds = the test dataset (if you have 3 folders)
  - class_names
  - output_folder: the path to save the pic
  - name
  '''  
  print("classification report: "+name, "\n",
        classification_report(y_pred , y_target, target_names=class_names)
        )

  
  clf_report = classification_report(y_pred , y_target, target_names=class_names, output_dict=True )
  plt.figure(figsize = (10,8))
  plt.title("classification report: "+name, fontsize=12)
  sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True)
  plt.savefig(output_folder+name+"-classification_report.png")
  # plt.show()

def confusion_matrix_report(y_pred, y_target,  class_names, output_folder, name):
  '''
  displays inline a pic and also saves it
  - y_pred: inferences
  - y_target: labels
  - class_names
  - output_folder: path to save pic
  - name
  '''
  
  # confusion matrix without normalizating results
  cm_NOTnorm = confusion_matrix(y_target, y_pred, normalize=None ) #,  labels=[class_names])
  df_cm_NOTnorm = pd.DataFrame(cm_NOTnorm, index= [class_names], columns = [class_names])
  fig = plt.figure(figsize = (10,8))
  ax1 = fig.add_subplot(1,1,1)
  sns.set(font_scale=1.4) #for label size
  sns.heatmap(df_cm_NOTnorm, annot=True, annot_kws={"size": 11},fmt='.1f') # fmt removes scientific notation 1 decimal
  plt.title("confusion matrix NOT norm: "+name, fontsize=10)
  ax1.set_ylabel('True Values',fontsize=14)
  ax1.set_xlabel('Predicted Values',fontsize=14)
  plt.savefig(output_folder+name+"-confusion_matrix_NOTnorm.png")
  plt.show()
  
  # confusion matrix normalized
  cm_norm = confusion_matrix(y_target, y_pred, normalize="true" ) #,  labels=[class_names])
  df_cm_norm = pd.DataFrame(cm_norm, index= [class_names], columns = [class_names])
  fig = plt.figure(figsize = (10,8))
  ax1 = fig.add_subplot(1,1,1)
  sns.set(font_scale=1.4) #for label size
  sns.heatmap(df_cm_norm, annot=True, annot_kws={"size": 12})
  plt.title("confusion matrix NORM: "+name, fontsize=10)
  ax1.set_ylabel('True Values',fontsize=14)
  ax1.set_xlabel('Predicted Values',fontsize=14)
  plt.savefig(output_folder+name+"-confusion_matrix_NORM.png")
  plt.show()
  
  