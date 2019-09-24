#!/usr/bin/env python
# coding: utf-8

# In[56]:


# Description: This program classified the MNIST handwritten digit images
#               as a number 0-9


# In[37]:


get_ipython().system('pip install tensorflow keras numpy mnist matplotlib')


# In[38]:


# Import the packages / dependencies

import numpy as np
import mnist # Get data set from
import matplotlib.pyplot as plt # Graph
from keras.models import Sequential # ANN architecture
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D # The layers in the ANN
from keras.utils import to_categorical


# In[39]:


'''
Another way to access MNIST dataset is through Tensorflow and keras.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
'''

# Load the data set
train_images = mnist.train_images() # Training data images
train_labels = mnist.train_labels() # Training data labels

test_images = mnist.test_images() # Testing data images
test_labels = mnist.test_labels() # Testing data labels


# In[40]:


image_index = 100
plt.imshow(train_images[image_index], cmap='Greys')
print(train_labels[100])


# In[41]:


# Reshaping the array to 4-dims so that it can work with the Keras API
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1) 
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)  

# Making sure that the values are float so that we can get decimal points
train_images = train_images.astype('float32')
test_images = test_images.astype('float32')

# Normalize the images. Normalize the pixel values from [0, 255] to
#  [0, 1]. This is always required for neural network models.
train_images /= 255
test_images /= 255

# Print the shape
print(train_images.shape) # 60,000 rows and 784 columns
print(test_images.shape)  # 10,000 rows and 784 columns


# In[42]:


# Build the model
# 3 layers --> 2 layers with 64 neurons and the relu function +
#              1 layer with 10 neurons and softman function

model = Sequential()
model.add(Conv2D(28, kernel_size=(3,3), input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten()) # Flattening the 2D arrays for fully connected layers

'''
If the model doesn't need the above layers, then simply reshape the 
train_images and test_images to a 2D array.
'''  

model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation = 'softmax'))


# In[43]:


# Compile the model

'''
The loss function measures how well the model did on train, and 
then tries to improve on it using the optimizer.

If loss='categorical_accuracy' then will need to hot_encode the
train_labels and test_labels
'''

model.compile(
    optimizer='adam', loss = 'sparse_categorical_crossentropy', # not sparse_categorical_crossentropy since Mutual Exclusive Data
    metrics = ['sparse_categorical_accuracy'] # If metrics='accuracy' then sparse will be automatically chosen
             )


# In[44]:


len(train_labels)


# In[45]:


# Train the model
model.fit(
    train_images, train_labels, epochs = 5
)


# In[46]:


# Evaluate the code
model.evaluate(test_images, test_labels)


# In[47]:


# Predict on the first 5 test images. 
predictions = model.predict(test_images[:5])

# Print our models predictions
print(np.argmax(predictions, axis=1)) # Max prob from a row
print(test_labels[:5])


# In[48]:


for i in range(0,5):
    first_image = test_images[i]
    print(type(first_image))
    #first_image = np.array(first_image, dtype='float')
    pixels = first_image.reshape((28,28))
    plt.imshow(pixels)
    plt.show()


# In[ ]:




