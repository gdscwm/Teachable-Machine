# Teachable-Machine
training a machine learning model for image classification


PowerPoint Explanation!

Go to: https://teachablemachine.withgoogle.com/

Download the dataset: https://www.microsoft.com/en-us/download/details.aspx?id=54765 (or get your own pictures)

Install VS Code or some form of Python IDE. 


Write the model code:


To install Tensorflow, run:

```
pip install tensorflow
```

We will start with imports and loading in the model:

```
from tensorflow import keras
import cv2
import numpy as np


# Load the model
model = keras.models.load_model("keras_model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

image_path = "/PetImages/Cat/1.jpg" #add the path of your image directory 
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

Yay!!! You have created an image classifier! ML!


Further Work:

You can implement a heatmap visualization on the image:



