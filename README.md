# Final Master's Work at VIU.


----------------------------------

## kfolds_validation branch:

Using the data I have to create 5 different physical folders where all the data is randomly shuffled each time. Then, a model will be tested in each one. The data is batch-loaded from each folder using the TensorFlow method image_dataset_from_directory. Created to check:

- The goodness of the shuffled data (regular folders section)
- How lack of data affects model performance (irregular folders section)




## Methodology.

Assuming that train/validation datasets are used to check model performance, and test dataset to choose between models and fine-tuning:

- Take 150 pics of each class for the test dataset. These pics will never be merged with the others again.

With the rest of pictures:
- Regular folders:
    - 5 folders for train/validation datasets will be created.
    - They will have shuffled data (so the 5 folders will have different distribution of pics in train/validation datasets).
    
- Irregular folders:
    - Create 4 folders with each time less pictures in the training set, to get the study of how lack of input data affects the model.

