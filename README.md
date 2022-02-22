# Final Master's Work at VIU.


<details>
  <summary>Main branch intro.</summary>
 This is a Computer Vision classification for Real State. The intention of this project is to improve the classification of different spaces inside houses. 

Consisting of:
- Fine-tuning of pre-trained models.
- Data augmentation.
    - TensorFlow method *ImageDataGenerator* in pickled data.
    - Adding data augmentation as a sequential layer using the TensorFlow method *image_dataset_from_directory*  straight from the pics folder.
- Study the creation of synthetic data.
</details>



----------------------------------

## kfolds_validation branch:

Using the data I have to create 5 different physical folders where all the data is randomly shuffled each time. Then, a model will be tested in each one. The data is batch-loaded from each folder using the TensorFlow method image_dataset_from_directory. Created to check:

- The goodness of the shuffled data (regular folders section)
- How lack of data affects model performance (irregular folders section)

----------------------------------


## 1. Methodology.

Assuming that train/test datasets are used to check model performance, and validation dataset to choose between models and fine-tuning (I believe the right interpretation is with val and test the other way around):

- Take 150 pics of each class for the validation dataset. These pics will never be merged with the others again.

With the rest of pictures:
- Regular folders:
    - 5 folders for train/test datasets will be created.
    - They will have shuffled data (so the 5 folders will have different pics in train/test datasets).
    
- Irregular folders:
    - Create 4 folders with each time less pictures in the training set, to get the study of how lack of input data affects the model.

## 2. Development.

- The creation of this different subfolders can be found in src/creating_5_subfolders_for_kfoldslike_validation 
- How to test a neural network architecture against them can be found in src/1val_5traintest_folders

## 3. Results.

## 4. Conclusions.

![alt](output/catstruction.png)