# Final Master's Work at VIU.

*This repository is intened to just reinforce some aspests of the PDF thesis.*


Nowadays we have the tools and technology to create powerful and efficient models of Artificial Intelligence for every field in which we want to work, for every task we desire to improve.

For some fields of AI, these models usually need a high volume of input data in order to work properly. Unfortunately, obtaining the required amount is in most of the cases the biggest challenge to overcome.

In this final project, which is a Classification problem of house spaces through Computer Vision:
- Different approaches are evaluated to avoid overfitting and improve metrics, while working with a limited, unbalanced dataset.
- Several pre-trained models with different configurations were adapted to our purpose. 
- Different methods of data ingestion were studied. 
- Custom data-augmentation layers were tested as well. 
- Synthetic data was created for always-misclassified pictures, in an attempt to improve results after increasing the volume of those kinds of images.

---------------------------

# Introduction.
<details>
  <summary>Click to expand.</summary>


The aim of this Master’s Final Project is to work with a reduced dataset in order to improve the performance of an image classifier system for different spaces inside houses.

Landing the idea properly, a journey through ML tools and Data Augmentation techniques will be started. Every time a dead-end will be reached, there will be countless alternatives already available in the open science world. Without hesitation, sometimes the real challenge will be to make a decision about which one of these technologies should be the next step.


Here it is presented a brief introduction for the performed stages explained in Section Methodology:

1. The starting point will not be the most interesting academically, yet the most important one: The dataset.     

2. Different Data ingestion methods to feed our Neural Networks will be tested. 
They will be divided in 2 groups: The ones working with the whole dataset loaded in the RAM and the ones implemented as first layers of Neural Network architectures, that will process data in batch-processing.

3. A set of Neural Networks models will be chosen to work with. 
Neural Networks are a simplified model of the way a brain processes information. They consist of neurons, the basic units, arranged in layers. The model can be divided into the input layer (units representing the input fields), the hidden layers and the output layer (one or more units representing the target fields). 
The units are connected with varying connection strengths or weights. While training, initially those weights are random and are propagated from each neuron to every neuron in the next layer until reaching the output layer, generating an inference. Once done, those inferences are evaluated. This iteration is called an epoch, and it is repeated many times. Each time, weights are re-adjusted to get a better outcome (IBM Corporation, n.d.).
The mentioned representation of layers is usually presented in a more pragmatic fashion: The base model (input and hidden layers already trained by the provider), and the top model (hidden and output layers that will be trained by ourselves with our set of data). 
This process is called transfer-learning.
In this project, several configurations of the same architectures, but with different parameters will be tested in Section Architectures and Fine-tuning.

4. Due to our lack of data, different methods of Data Augmentation will be tested in order to improve the heterogeneity of our dataset, and also its volume. 

5. After building Neural Network architectures (from now on, NN) with different hyperparameters, different methods for gathering and augmentate data, the performance of a set of models will be tested. These are our trained Models.

6. Before running the models, our dataset will be shuffled and divided at first into training and validation sets (later train-validation-tests). 
We realised that the performance of models was highly dependent on some kind of images. If data was shuffled and models were run again, differences in performance may be big enough to not ignore this fact. For this reason the Physical k-folds validation will be implemented.
The sklearn method of cross_validation() (Scikit-learn, n.d.)  reads the available data, shuffles it and creates as many folders as wanted. Those folders are not persisted, they are just RAM-stored. Its purpose is to run a model in k different sets of data and measure the model performance as an average of the resulting metrics for each folder.
The same idea will be reproduced. Several folders distributions will be created, this time persisted in disk. Some of the distributions will be meant to measure the goodness of data, other distributions will be set to study the performance of a model over a decreasing-in-volume set of data. 

7. Trying to improve our results, over one of these distributions of folders we will perform inferences, in order to get exactly what images are always misclassified by all models. Those images will be located, and synthetic data will be created using them as input. Results will be presented in the Synthetic data section. This synth data will be created using an available Pytorch repository (GitHub Zhao et al., 2020). 
Once the non-real images are created, several ways to merge them with the real data will be tested. Finally, a model will be trained over different distributions of real plus unreal pictures.



</details>




----------------------------------

The sections of this Master's thesis can be found in the different branches of the project. 


- **pickle_input.**

  - How to create a pickle from your input data, load it and use data augmentation in memory as a whole.
  - Advantages and disadvantages.

- **trainvaltest_split.**

  - First testing of all models, using just the train and validation set.

- **kfolds_validation.**

  - Use the data available to create 5 different physical folders where all the images are randomly shuffled each time. Then, a model will be tested in each one. The data is batch-loaded from each folder using the TensorFlow method *image_dataset_from_directory*. Created to check:
  - The goodness of the shuffled data (regular folders section)
  - How lack of data affects model performance (irregular folders section)

- **synthetic_data_study.**
  - For all regular folders mentioned above, target what images of each class are always misclassified by every model.
  - For that images, generate synthetic data of them.

- **synthetic_data_application.**
  - For the regular folders mentioned in the *kfolds_validation* section, add 50, 250 and 480 synthetic pics for each class. Evaluate models performance.


- Results soon available in PDF format.


----------------------------------

**Personal info**
<details>
  <summary>Click to expand.</summary>

https://www.linkedin.com/in/alberto-vargas-pina/

![alt](output/science_dog.webp)

</details>
