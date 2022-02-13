# Final Master's Work at VIU.
## Computer Vision classification for Real State.

----------------------------------


Hi there! As brief introduction, this project consist of:
- Computer Vision to classify different spaces inside houses.
- Improve pre-trained models for that purpose.

----------------------------------

#### Further description:
<details>
  <summary>Click to expand</summary>

- How to save datasets of images in a pickle and work with it.
- Data augmentation for pickled data.
- Pickling data VS batch loading them straight from the folder.
- Using the tensorflow **image_dataset_from_directory** method.
    - Comparing between using this method straight to the dataset or after creating the *train, test, validation* folders manually.
    - How to create the *train, test, validation* folders shuffling data in order to have a kind of stratified k-folds validation, with physically different folders.
- Data augmentation layer added to sequential nn architectures.
- Generating synthetic data with StyleGAN
- Goodness of different pre-trained models with fine-tuning

</details>


------------------------------------------------

##### Resources:
<details>
  <summary>Click to expand</summary>
    
The dataset I'm working with: House room dataset (5 different rooms)
    - bathroom
    - bedroom
    - dinning room
    - kitchen
    - living room
    - https://www.kaggle.com/robinreni/house-rooms-image-dataset
    
    
Other possible resources:

- Dataset: House price prediction with exterior front of houses (socal)
    - front (pics)
    - street
    - city
    - number
    - number of bedrooms
    - nomber of bathrooms
    - square feet
    - price
    - https://github.com/ted2020/House-Price-Prediction-via-Computer-Vision (no sé cómo usarlo)
    - kaggle dataset => https://www.kaggle.com/ted8080/house-prices-and-images-socal

- Dataset: House price prediction with 4 different rooms:
    - bathroom  (pics)
    - bedroom    (pics)
    - kitchen    (pics)
    - front of the house     (pics)
    - number of bedrooms
    - number of badooms
    - area
    - zipcode
    - price
    - https://github.com/emanhamed/Houses-dataset (paper incl)


- RoomNet CNN. A Convolutional Neural Net to classify pictures of different rooms of a house/apartment (i don't know how to use it). Trained to classify 6 classes:
    - backyard
    - bathroom
    - bedroom
    - frontyard
    - kitchen
    - livingRoom
    - https://towardsdatascience.com/fast-real-estate-image-classification-using-machine-learning-with-code-32e0539eab96
    - 

- EXAMPLE: Monk library for house room type classification (7 classes)
    - Exterior 
    - bedroom
    - kitchen
    - living_room
    - Interior
    - bathroom
    - dining_room
    - https://towardsdatascience.com/image-classifier-house-room-type-classification-using-monk-library-d633795a42ef
    - https://github.com/Tessellate-Imaging/monk_v1/blob/master/study_roadmaps/4_image_classification_zoo/Classifier%20-%20House%20room%20type%20Claasification.ipynb

- categorizing listing airbnb photos (vigulgativo, no tiene dataset)
    -  Bedrooms
    - Bathrooms
    - Living Rooms
    - Kitchens
    - Swimming Pools
    - Views.
    - https://medium.com/airbnb-engineering/categorizing-listing-photos-at-airbnb-f9483f3ab7e3

</details>

----------------------------------------------


**Personal info**
<details>
  <summary>Click to expand.</summary>

https://www.linkedin.com/in/alberto-vargas-pina/

![alt](output/science_dog.webp)

</details>
