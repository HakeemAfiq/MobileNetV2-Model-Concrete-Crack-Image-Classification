#%%
# Import packages
import os
import numpy as np
import matplotlib.pyplot as plt
import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, applications, optimizers, losses, callbacks

from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix

#%%
# Data loading
PATH = os.path.join(os.getcwd(), 'Dataset', 'Concrete Crack Images for Classification')

# %%
# Data preparation
# Define the batch size and image size
SEED = 64
BATCH_SIZE = 32
IMG_SIZE = (160,160)

# Load the data into tensorflow dataset
# Split the dataset into train (75%) and validation (25%) dataset
train_dataset = keras.utils.image_dataset_from_directory(PATH, shuffle=True, validation_split=0.25, subset='training', seed=SEED, batch_size=BATCH_SIZE, image_size=IMG_SIZE)
val_dataset = keras.utils.image_dataset_from_directory(PATH, shuffle=True, validation_split=0.25, subset='validation', seed=SEED, batch_size=BATCH_SIZE, image_size=IMG_SIZE)

# %%
# Display some images as example
class_names = train_dataset.class_names

plt.figure(figsize=(10,10))
for images,labels in train_dataset.take(1):
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.imshow(images[i].numpy().astype('uint8'))
        plt.title(class_names[labels[i]])
        plt.axis('off')

# %%
# Split the validation dataset into validation and test dataset
val_batches = tf.data.experimental.cardinality(val_dataset)
test_dataset = val_dataset.take(val_batches//5)
validation_dataset = val_dataset.skip(val_batches//5)

# %%
# Convert the BatchDataset into PrefetchDataset
AUTOTUNE = tf.data.AUTOTUNE

prefetch_train = train_dataset.prefetch(buffer_size=AUTOTUNE)
prefetch_val = validation_dataset.prefetch(buffer_size=AUTOTUNE)
prefetch_test = test_dataset.prefetch(buffer_size=AUTOTUNE)

# %%
# Create a small pipeline for data augmentation
data_augmentation = keras.Sequential()
data_augmentation.add(layers.RandomFlip('horizontal'))
data_augmentation.add(layers.RandomRotation(0.2))

# %%
#Apply the data augmentation and display the image for inspection
for images,labels in prefetch_train.take(1):
    first_image = images[0]
    plt.figure(figsize=(10,10))
    for i in range(9):
        plt.subplot(3,3,i+1)
        augmented_image = data_augmentation(tf.expand_dims(first_image,axis=0))
        plt.imshow(augmented_image[0]/255.0)
        plt.axis('off')

# %%
# Prepare the layer for data preprocessing
preprocess_input = applications.mobilenet_v2.preprocess_input

# Apply transfer learning
IMG_SHAPE = IMG_SIZE + (3,)
feature_extractor = applications.MobileNetV2(input_shape=IMG_SHAPE,include_top=False,weights='imagenet')

# Disable the training for the feature extractor (freeze the layers)
feature_extractor.trainable = False
feature_extractor.summary()
keras.utils.plot_model(feature_extractor,show_shapes=True)

# %%
# create the classification layers
global_avg = layers.GlobalAveragePooling2D()
# l1 = keras.regularizers.L1()
output_layer = layers.Dense(len(class_names), activation='softmax')

#%%
# Use functional API to link all of the modules together
inputs = keras.Input(shape=IMG_SHAPE)
x = data_augmentation(inputs)
x = preprocess_input(x)
x = feature_extractor(x)
x = global_avg(x)
x = layers.Dropout(0.3)(x)
outputs = output_layer(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()

# %%
# Model compilation
optimizer = optimizers.Adam(learning_rate=0.0001)
loss = losses.SparseCategoricalCrossentropy()
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

#%%
# Model evaluation before model training
loss0, accuracy0 = model.evaluate(prefetch_val)
print('Loss = ', loss0)
print('Accuracy = ', accuracy0)

#%%
# Tensorboard callback
es = keras.callbacks.EarlyStopping(patience=2,verbose=1,restore_best_weights=True)
log_path = os.path.join('log_dir', 'Concrete_cracks_image_classification', datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
tb = callbacks.TensorBoard(log_dir=log_path)

#%%
# Model training
EPOCHS = 15
history = model.fit(prefetch_train, validation_data=prefetch_val, epochs=EPOCHS, callbacks=[tb,es])

#%%
# Model evaluation
test_loss,test_acc = model.evaluate(prefetch_test)

print("Loss = ",test_loss)
print("Accuracy = ",test_acc)
# %%
#Deploy the model using the test data
image_batch, label_batch = prefetch_test.as_numpy_iterator().next()
predictions = np.argmax(model.predict(image_batch),axis=1)

#Compare label and prediction
label_vs_prediction = np.transpose(np.vstack((label_batch,predictions)))

# %%
# Model Analysis
# confusion matrix and classification report of model prediction
print(classification_report(label_batch, predictions))
cm = confusion_matrix(label_batch, predictions)
ConfusionMatrixDisplay(cm)

# %%
# Plot the confusion matrix
disp = ConfusionMatrixDisplay(cm)
disp.plot()

# %%
# Model saving
model.save('Concrete_cracks_image_classification_model.h5')

# %%
