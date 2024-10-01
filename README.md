# EX 04 IMAGE-TRANSFORMATIONS

## Aim
To perform image transformation such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping using OpenCV and Python.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
### Step 1: 
Import numpy module as np and pandas as pd.
### Step 2: 
Assign the values to variables in the program.
### Step 3: 
Get the values from the user appropriately.
### Step 4: 
Continue the program by implementing the codes of required topics.
### Step 5: 
Thus the program is executed in google colab.

## Program:

#### Developed By : Sriram G
#### Register Number : 212222230149

### Installing OpenCV , importing necessary libraries and displaying images  

```
# Install OpenCV library
!pip install opencv-python-headless

import cv2
import numpy as np
from matplotlib import pyplot as plt
```

#### (i) Image Translation

```
# Load an image from URL or file path
image_url = 'SPB.jpeg'  
image = cv2.imread(image_url)

# Define translation matrix
tx = 50  # Translation along x-axis
ty = 30  # Translation along y-axis
translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])  # Create translation matrix

# Apply translation to the image
translated_image = cv2.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]))

# Display original and translated images
print("Original Image:")
plt.figure(figsize=(6, 6))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
print("Translated Image:")
plt.figure(figsize=(6, 6))
plt.imshow(cv2.cvtColor(translated_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
```

#### (ii) Image Scaling

```
# Load an image from URL or file path
image_url = 'SPB.jpeg'  # Replace with your image URL or file path
image = cv2.imread(image_url)


# Define scale factors
scale_x = 1.5  # Scaling factor along x-axis
scale_y = 1.5  # Scaling factor along y-axis


# Apply scaling to the image
scaled_image = cv2.resize(image, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)

# Display original and scaled images
print("Original Image:")
plt.figure(figsize=(6, 6))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
print("Scaled Image:")
plt.figure(figsize=(6, 6))
plt.imshow(cv2.cvtColor(scaled_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

```

#### (iii) Image shearing
```
# Load an image from URL or file path
image_url = 'SPB.jpeg'  # Replace with your image URL or file path
image = cv2.imread(image_url)

# Define shear parameters
shear_factor_x = 0.5  # Shear factor along x-axis
shear_factor_y = 0.2  # Shear factor along y-axis

# Define shear matrix
shear_matrix = np.float32([[1, shear_factor_x, 0], [shear_factor_y, 1, 0]])

# Apply shear to the image
sheared_image = cv2.warpAffine(image, shear_matrix, (image.shape[1], image.shape[0]))

# Display original and sheared images
print("Original Image:")
plt.figure(figsize=(6, 6))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
print("Sheared Image:")
plt.figure(figsize=(6, 6))
plt.imshow(cv2.cvtColor(sheared_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
```
#### (iv) Image Reflection

##### Horizontal Reflection:

```
# Load an image from URL or file path
image_url = 'SPB.jpeg'  # Replace with your image URL or file path
image = cv2.imread(image_url)
print("Original Image:")
plt.figure(figsize=(6, 6))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
# Reflect the image horizontally
print("↑ Reflected Horizontally")
reflected_image_horizontal = cv2.flip(image, 1)
plt.figure(figsize=(6, 6))
plt.imshow(cv2.cvtColor(reflected_image_horizontal, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
```
##### Vertical Reflection:
```
print("Original Image:")
plt.figure(figsize=(6, 6))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
# Reflect the image vertically
print("↑ Reflected Vertically")
reflected_image_vertical = cv2.flip(image, 0)
plt.figure(figsize=(6, 6))
plt.imshow(cv2.cvtColor(reflected_image_vertical, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
```
##### Both Horizontal and Vertical Reflection:

```
print("Original Image:")
plt.figure(figsize=(6, 6))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
# Reflect the image both horizontally and vertically
print("↑ Reflected Both")
reflected_image_both = cv2.flip(image, -1)
plt.figure(figsize=(6, 6))
plt.imshow(cv2.cvtColor(reflected_image_vertical, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
```

### (v) Image Rotation

```
# Load an image from URL or file path
image_url = 'SPB.jpeg'  # Replace with your image URL or file path
image = cv2.imread(image_url)


# Define rotation angle in degrees
angle = 45

# Get image height and width
height, width = image.shape[:2]

# Calculate rotation matrix
rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)

# Perform image rotation
rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

# Display original and rotated images
print("Original Image:")
plt.figure(figsize=(6, 6))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
print("Rotated Image:")
plt.figure(figsize=(6, 6))
plt.imshow(cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
```

### (vi) Image Cropping

```
# Load an image from URL or file path
image_url = 'SPB.jpeg'  # Replace with your image URL or file path
image = cv2.imread(image_url)

# Define cropping coordinates (x, y, width, height)
x = 100  # Starting x-coordinate
y = 50   # Starting y-coordinate
width = 200  # Width of the cropped region
height = 150  # Height of the cropped region

# Perform image cropping
cropped_image = image[y:y+height, x:x+width]

# Display original and cropped images
print("Original Image:")
plt.figure(figsize=(6, 6))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
print("Cropped Image:")
plt.figure(figsize=(6, 6))
plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
```
## Output:
### i)Image Translation
<br>
<br>
<br>
<br>

### ii) Image Scaling
<br>
<br>
<br>
<br>


### iii)Image shearing
<br>
<br>
<br>
<br>


### iv)Image Reflection
<br>
<br>
<br>
<br>



### v)Image Rotation
<br>
<br>
<br>
<br>



### vi)Image Cropping
<br>
<br>
<br>
<br>




## Result: 

Thus the different image transformations such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping are done using OpenCV and python programming.
