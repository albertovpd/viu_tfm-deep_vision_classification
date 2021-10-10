# Final Master's Work at VIU.
## Computer Vision classification for Real State.

----------------------------------


Hi there!

This repo is going to be quite messy while finding the best approach to the goals, which at the moment, look like this:
- Computer Vision to classify different spaces inside houses.
- Improve pre-trained models for that purpose.
- Study the viability of creating a Real State pricing problem, adding the previous results as features.

-------------------------
## Updates & evolution.

- https://docs.google.com/document/d/1z8G_THiNSHe14UcoJhIIcJL4NLlCuOiVTu6yvla37qo/edit


# Branches

- vgg19-analyticsindia
    - overfitting
    - too much layers
    - trained with too much classes

- vgg16

- resnet34 (if available)/resnet50 

-------------------------------------------
-------------------------------------------

- Tensorflow-GPU on Colab => https://colab.research.google.com/notebooks/gpu.ipynb#scrollTo=oM_8ELnJq_wd
- Why using GPU for Deep Learning => https://towardsdatascience.com/what-is-a-gpu-and-do-you-need-one-in-deep-learning-718b9597aa0d

-------------------------

Developer stuff:
<details>
  <summary>Click to expand</summary>

- https://github.com/users/albertovpd/projects/8
- https://github.com/albertovpd/viu-fundamentos_redes_neuronales/tree/main/colabs/tasks
- https://github.com/albertovpd/viu-fundamentos_redes_neuronales/blob/main/colabs/clase_2y3-Introduccion_a_Deep_Learning_Colab_ipynb.ipynb

- https://analyticsindiamag.com/transfer-learning-for-multi-class-image-classification-using-deep-convolutional-neural-network/
- https://machinelearningmastery.com/learning-curves-for-diagnosing-machine-learning-model-performance/
- https://towardsdatascience.com/handling-overfitting-in-deep-learning-models-c760ee047c6e
- https://www.kaggle.com/ryanholbrook/overfitting-and-underfitting

</details>



------------------------------------

### Roadmap: 
<details>
  <summary>Click to expand</summary>

https://github.com/users/albertovpd/projects/8

- Starting from here => https://www.kaggle.com/robinreni/house-rooms-image-dataset
- Inspect dataset
- Basic NN start:
    - Alexnet
    - ResNet
    - SqueezeNet
    - EfficientNet
    - any other I think it could be great
- Monk testing

</details>

-----------------------------------------------

Stuff to check:
<details>
  <summary>Click to expand</summary>

De momento puedes ir echándole un ojo a estos enlaces. Si alguno no puedes porque es de pago, puedes usar sci-hub:


    - https://towardsdatascience.com/fast-real-estate-image-classification-using-machine-learning-with-code-32e0539eab96
    - https://www.researchgate.net/publication/316494092_Real_Estate_Image_Classification
    - https://restb.ai
    - https://vize.ai/real-estate
    - https://www.spiedigitallibrary.org/conference-proceedings-of-spie/11069/110691I/Classification-of-real-estate-images-using-transfer-learning/10.1117/12.2524417.short?SSO=1
    - https://www.trulia.com/blog/tech/image-recognition/
    - https://datafiniti.co/products/property-data/?gclid=Cj0KCQjwjPaCBhDkARIsAISZN7QOjbchXQJ_mQ37hpHtyPSpK-AV7S-LpZ-BxvjY2ic4vr3oxRlacWkaAjSXEALw_wcB
    - https://ieeexplore.ieee.org/abstract/document/7926631
    
</details>

------------------------------------------------

### Pending sources at this time:
<details>
  <summary>Click to expand</summary>

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

- Dataset: House room dataset (5 different rooms)
    - bathroom
    - bedroom
    - dinning room
    - kitchen
    - living room
    - https://www.kaggle.com/robinreni/house-rooms-image-dataset


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








-----------------------------------

#### Notes.
<details>
  <summary>Click to expand.</summary>
TFM: Línea Temática: Sistema automático de clasificación de viviendas mediante el uso únicamente de fotografías


# Notes

- RoomNet
    - forkeada
    - 6 clasificaciones: Backyard-0, Bathroom-1, Bedroom-2, Frontyard-3, Kitchen-4, LivingRoom-5
- Redes neuronales convolucionales / versiones, como la vgg-19
- long short term memory (LSTM), and fully connected neural networks

- Preprocesado: 'contrast-limited adaptive histogram equalization (CLAHE) for image enhancement
- AHE: es una manera de mejorar el contraste en imágenes, con una superposición de histogramas, cada uno correspondiendo a partes diferentes de la imagen. Mejora los bordes y la definición, pero puede meter mucho ruido en zonas homogéneas de la imagen, y CLAHE lo qu ehace es optimizar esto.

- https://towardsdatascience.com/fast-real-estate-image-classification-using-machine-learning-with-code-32e0539eab96

- https://www.researchgate.net/publication/316494092_Real_Estate_Image_Classification

- https://restb.ai/
    - no veo que haya una api disponible para uso personal gratuito

- https://vize.ai/real-estate
    - no veo que haya una api disponible para uso personal

- https://www.spiedigitallibrary.org/conference-proceedings-of-spie/11069/110691I/Classification-of-real-estate-images-using-transfer-learning/10.1117/12.2524417.short?SSO=1 
    - de pago

- https://www.trulia.com/blog/tech/image-recognition/#
    - entiendo que esta web es para coger ideas

- https://datafiniti.co/products/property-data/?gclid=Cj0KCQjwjPaCBhDkARIsAISZN7QOjbchXQJ_mQ37hpHtyPSpK-AV7S-LpZ-BxvjY2ic4vr3oxRlacWkaAjSXEALw_wcB
    - lo mismo pero tiene ubicación en google maps

- https://ieeexplore.ieee.org/abstract/document/7926631
    - de pago

# PASOS:

- descargar datasets públicos
    - con imágenes
    - buscar dataset con precio, e imágenes, 500-600 imágenes

- usar la roomNet para clasificar automáticamente las imágenes... o no

-  keras
    -  https://customers.pyimagesearch.com/lesson-sample-training-your-first-cnn/
    - https://www.pyimagesearch.com/2018/04/16/keras-and-convolutional-neural-networks-cnns/


</details>

---------------

**Personal info**
<details>
  <summary>Click to expand.</summary>

https://www.linkedin.com/in/alberto-vargas-pina/

![alt](output/science_dog.webp)

</details>
