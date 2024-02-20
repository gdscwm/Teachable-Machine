# Teachable-Machine


Training a machine learning model for binary image classification

PowerPoint Explanation!


## Setup:

Go to: https://teachablemachine.withgoogle.com/

Download and unzip the dataset: https://www.microsoft.com/en-us/download/details.aspx?id=54765 (or get your own pictures)

Install VS Code or some form of Python IDE. 
VS Code: https://code.visualstudio.com/

PyCharm: https://www.jetbrains.com/pycharm/download/

## Teachable Machine Model Creation:

Go to Teachable Machine. Select Get Started.

Create a new Image Project -> Standard image model

Upload images into class 1 and class 2. (You can use our dataset images or upload your own, there is also a webcam feature)

In our example, we are using Dogs are Class 1 and Cats as Class 2. 

After images are added, select train. 

You can make changes to the Epochs, batch size, and learning rate using the "Advanced" drop-down.

## Explanation 

We are creating a binary image classifier, which has one class set as '1' and the other set as '0'. 

Epochs determine how many times data is passed through the model. Normally more is better, but it depends on the size of the data set. Batch size is the set of samples that are used for training. 
If we use 100 images for each class and the default batch size is 16, then our batch size would be 100/16 or about 6. Learning rate is how hyperparameters influence the model learning speed. 


Look at the model output and test selecting different images. See how accurate the classification is. 

Select 'Export' model and download the Tensorflow Keras model. This converts the model to a keras .h5 model which you can further make challenges to with Python. 

Open up VS Code or another Python IDE. Open the model and data set in the same directory. 


## Write the model code:

Create a python file and name it model.py

Important: make sure the dataset, python file, and keras.h5 model are in the same directory/Python project. 

To install Tensorflow, run:

```
pip install tensorflow
```

TensorFlow is a library used for machine learning predictions. 


If you get an error about opencv, you can install it by running: 
```
pip install opencv-python
```
Opencv is used for image processing. 


We will start with imports and loading in the model:

```
from tensorflow import keras
import cv2
import numpy as np


# Load the model
model = keras.models.load_model("keras_model.h5", compile=False)

# Load the labels (This is what classifies the images as either a cat or dog)
class_names = open("labels.txt", "r").readlines()

image_path = "/PetImages/(Cat or Dog)/ImageNumber.jpg" #add the path of your image directory
image = cv2.imread(image_path)
```

Resize the image for prediction and performing preprocessing:

```
# Resize the image into (224-height,224-width) pixels
image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

# Make the image a numpy array and reshape it to the model's input shape
image_input = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

# Normalize the image array
image_input = (image_input / 127.5) - 1
```

Load in the model and predictions: 

```
# Predict the model
prediction = model.predict(image_input)
index = np.argmax(prediction)
class_name = class_names[index]
confidence_score = prediction[0][index]
```
Display classification 

```
#Get the class accuracy an confidence to display on the label 
text = "Class: {} Confidence: {:.2f}% ".format(class_name[2:], confidence_score * 100)

# Determine the width of the widest line of text
text_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][0]

# Determine the width of the image
image_width = image.shape[1]

# Determine the width of the label (take the maximum of text_width and image_width)
label_width = max(text_width, image_width)

# Create a blank image for the label with a different height
label_height = 50
label = np.zeros((label_height, label_width, 3), dtype=np.uint8)
cv2.putText(label, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# Make the image wider
image = cv2.resize(image, (label_width, image.shape[0]), interpolation=cv2.INTER_AREA)

# Concatenate the original image and the label vertically
concatenated_image = np.vstack((image, label))
```

Display the window: 

```


# Create a resizable window
cv2.namedWindow("Classified Image", cv2.WINDOW_NORMAL)

# Show the concatenated image with the classification label
cv2.imshow("Classified Image", concatenated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


```
This should load the image with the classification label into a new window. 


Yay!!! You have created an image classifier! ML!

## Further Exploration 

Some Resources:

Tensorflow basic image classification: https://www.tensorflow.org/tutorials/images/classification


You can try loading different images into the model by changing the image path:

```
image_path = "/PetImages/(Cat or Dog)/ImageNumber.jpg" #add the path of your image directory
```

You can generate an image heatmap: 

To generate the heatmap, we compute the gradient of the output class score using the activations of the last convolutional layer. 
This gradient represents how much each pixel in the input image contributes to the final class score.

First install matplot lib:

```
python -m pip install -U matplotlib
```

Import matplot at the top of the script and add tensorflow:

```
import matplotlib.pyplot as plt
import tensorflow as tf 
```


Code for generation:

```
# Get the output tensor of the last convolutional layer
last_conv_layer = model.get_layer("conv2d_2")

# Generate class activation heatmap
with tf.GradientTape() as tape:
    grads = tape.gradient(prediction, last_conv_layer.output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

# The gradients obtained are pooled across spatial dimensions using global average pooling.

#These gradients are then combined with the activations of the last convolutional layer to produce a heatmap. T
#The heatmap highlights the regions of the image that had the highest influence on the model's prediction.

heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer.output), axis=-1)
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)

# Resize heatmap to match the image size
heatmap = cv2.resize(heatmap, (resized_image.shape[1], resized_image.shape[0]))
heatmap = np.uint8(255 * heatmap)

# Apply heatmap to the original image
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = cv2.addWeighted(resized_image, 0.6, heatmap, 0.4, 0)

# Display the original image, heatmap, and superimposed image
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
axs[0].set_title('Original Image')
axs[0].axis('off')
axs[1].imshow(heatmap)
axs[1].set_title('Heatmap')
axs[1].axis('off')
axs[2].imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
axs[2].set_title('Superimposed Image')
axs[2].axis('off')
plt.show()
```

