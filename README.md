# Final Master's Work at VIU.


 Hi there. 

 This is a Computer Vision classification for Real State. The intention of this project is to improve the classification of different spaces inside houses. 

Consisting of:
- Fine-tuning of pre-trained models.
- Data augmentation.
    - TensorFlow method *ImageDataGenerator* in pickled data.
    - Adding data augmentation as a sequential layer using the TensorFlow method *image_dataset_from_directory*  straight from the pics folder.
- Study the creation of synthetic data.

----------------------------------

![alt](output/catstruction.png)

----------------------------------

In the different branches can be found the different approaches/sections of this project. Current useful branches:

- **pickle_input**
    - how to create a pickle from your input data and load it in memory as a whole.
    - advantages and disadvantages.
- **kfolds_validation** => use the data I have to create 5 different physical folders where all tje data is randomly shuffled each time. Then, a model will be tested in each one. The data is batch-loaded from each folder using the TensorFlow method *image_dataset_from_directory*. Created to check:
    - the goodness of the shuffled data (regular folders section)
    - how lack of data affects model performance (irregular folders section)


----------------------------------



### Roadmap: 
<details>
  <summary>Click to expand</summary>

https://github.com/users/albertovpd/projects/8

</details>

----------------------------------

**Personal info**
<details>
  <summary>Click to expand.</summary>

https://www.linkedin.com/in/alberto-vargas-pina/

![alt](output/science_dog.webp)

</details>
