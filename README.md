# Final Master's Work at VIU.


 Hi there!

 This is a Computer Vision classification problem for Real State. The intention of this project is the study of models performance with lack of data.

Consisting of:
- Fine-tuning of pre-trained models.
- Data augmentation.
    - TensorFlow method *ImageDataGenerator* in pickled data.
    - Adding data augmentation as a sequential layer using the TensorFlow method *image_dataset_from_directory*  straight from the pics folder.
- Study the creation of synthetic data.
- Use of synthetic data.

----------------------------------

![alt](output/catstruction.png)

----------------------------------

In the different branches can be found the different approaches/sections of this project. Current useful branches:

- **pickle_input.**

  - How to create a pickle from your input data, load it and use data augmentation in memory as a whole.
  - advantages and disadvantages.

- **kfolds_validation.**

  - Use the data available to create 5 different physical folders where all the images are randomly shuffled each time. Then, a model will be tested in each one. The data is batch-loaded from each folder using the TensorFlow method *image_dataset_from_directory*. Created to check:
  - the goodness of the shuffled data (regular folders section)
  - how lack of data affects model performance (irregular folders section)

- **synthetic_data_study.**
  - For all regular folders mentioned above, target what images of each class are always misclassified by every model.
  - For that images, generate synthetic data of them.

- **synthetic_data_application.**
  - For the regular folders mentioned in the *kfolds_validation* section, add 50, 250 and 480 synthetic pics for each class. Evaluate models performance.

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
