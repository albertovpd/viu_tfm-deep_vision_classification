# Final Master's Work at VIU.
## Computer Vision classification for Real State.

----------------------------------


Hi there! As brief introduction, this project consist of:
- Computer Vision to classify different spaces inside houses.
- Improve pre-trained models for that purpose.

----------------------------------

In this branch I show: 
- How to create a pickle with your input data (notebook **src/pickling_data**)
- Use it to work with a NN architecture.

Problems:
- First of all, I want to include a data augmentation layer in the NN architecture. For that is best practices to load the pics straight from the folders with batch processingl
- Loading an entire pickle means using your RAM for storing all that info and leads to RAM shortage.