# Final Master's Work at VIU.
## Computer Vision classification for Real State.

----------------------------------


Hi there!

This repo is going to be quite messy while finding the best approach to the goals, which at the moment, look like this:
- Computer Vision to classify different spaces inside houses.
- Enhace pre-trained models for that purpose.
- Study the viability of creating a Real State pricing problem, adding the previous results as features.


- Possible sources at this time:
    - https://www.kaggle.com/ted8080/house-prices-and-images-socal
    - https://github.com/emanhamed/Houses-dataset (paper incl)


- Mentions:
    - @article{ahmed2016house, title={House price estimation from visual and textual features}, author={Ahmed, Eman and Moustafa, Mohamed}, journal={arXiv preprint arXiv:1609.08399}, year={2016} } => https://github.com/emanhamed/Houses-dataset
-----------------------------------------

#### Personal and boring notes.
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

