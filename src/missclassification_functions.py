
# load models 
import tensorflow as tf
from tensorflow.keras import Model

# numpy stuff
import numpy as np

#navigate through folders
import os

def inferences_target_list(model, data):
    '''
    returns 2 lists: inferences list, real labels
    '''
    # over train set fold1
    y_pred_float = model.predict(data)
    y_pred = np.argmax(y_pred_float, axis=1)

    # get real labels
    y_target = tf.concat([y for x, y in data], axis=0) 
    y_target
    print("lenght inferences and real labels: ", len(y_pred), len(y_target))
    return y_pred, y_target


def get_missclassified(y_pred, y_target):
  '''
  returns a list with the indexes of real labels that were missclassified
  '''
  missclassified = []
  for i, (pred, target) in enumerate(zip(y_pred, y_target.numpy().tolist())):
    if pred!=target:
      #print(i, pred, target)
      missclassified.append(i)
  print("total missclassified: ",len(missclassified))
  return missclassified

def get_list_of_files(dirName):
    '''
    create a list of file and sub directories names in the given directory
    found here => https://thispointer.com/python-how-to-get-list-of-files-in-directory-and-sub-directories/
    ''' 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + get_list_of_files(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles