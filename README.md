# MobileNetV2 Model Concrete Crack Image Classification

![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)

### Summary
This project is an image classification project of datasets that contains images of cracked concrete in the positive folder and images of concretes without cracks in the negative folder. This project serves as a practice to get a better understanding on developing an NLP predictions model. The goal is to get atleast a 90% prediction accuracy and which was achieved as can be seen in the classification report below. The problems faced in this project is when trying to split the datasets into training,  validation and testing datasets. The datasets was only distributed into the two folders mentioned before. To split it, tensorflow.keras.image_dataset_from_directory() was used. The architecture of the model is developed by implementing transfer learning of keras model MobileNetV2.

### Classification report
![Classification report](https://user-images.githubusercontent.com/121662880/211803217-8f7fb2a4-1f1a-4600-a4c5-ba4f995ce69d.PNG)

### Confusion matrix
![Confusion matrix](https://user-images.githubusercontent.com/121662880/211803303-01113afe-777f-43c9-93c1-9e2eb2215f00.png)

### Tensorboard accuracy graph
![Tensorboard accuracy graph](https://user-images.githubusercontent.com/121662880/211803781-cb566507-dbf4-49e4-9ff7-d766d545f151.PNG)

### Tensorboard loss fraph
![Tensorboard loss graph](https://user-images.githubusercontent.com/121662880/211803820-42f35008-0057-4f18-aaf7-1f87813fa449.PNG)

### Credits
The datasets was obtained from --> https://data.mendeley.com/datasets/5y9wdsg2zt/2

### Model architecture
![Concrete cracks image classification model](https://user-images.githubusercontent.com/121662880/211804001-9d503ce7-dd97-4b2e-9a68-4ef33cb98230.png)
