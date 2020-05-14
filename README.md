# Handwritten-Number-Detection

**Idea**: 
Recognises handwritten digits

**What**: 
- This program classified the MNIST handwritten digit images as a number 0-9

**How**:
- Uses Neural Network to recognise handwritten digits
- Built 3 layers. 2 layers with 64 neurons and the relu function. 1 layer with 10 neurons and softmax function
- Chose *sparse_categorical_crossentropy* as a loss function since Mutual Exclusive Data. 
- Chose *sparse_categorical_accuracy* as metric, since if *accuracy* were chosen, *sparse* loss function would have been 
  automatically chosen

**Difficulties**
- Reshaping the array to 4-dims so that it can work with the Keras API
- Making sure that the values are float so that we can get decimal points
- Normalize the pixel values from [0, 255] to [0, 1]. This is always required for neural network models.
- Building the Neural Network model with various layers
- Deciding whether to hot encode the labels, which was dependent on the loss function. 



