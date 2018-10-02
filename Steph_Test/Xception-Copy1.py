
# coding: utf-8

# In[1]:


# Dependencies
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import os
import numpy as np
import tensorflow as tf
import pandas as pd
import keras
from keras.preprocessing import image
from keras.applications.xception import (
    Xception, preprocess_input, decode_predictions)


# In[2]:


# Load the Xception model
# https://keras.io/applications/#xception
model = Xception(
    include_top=True,
    weights='imagenet')


# In[3]:


# Default Image Size for Xception
image_size = (299, 299)


# In[7]:


# Load the image and resize to the input dimensions that Xception
# was trained with
image_path = os.path.join("Images", "pup.jpg")
img = image.load_img(image_path, target_size=image_size)
plt.imshow(img)


# In[8]:


# Preprocess image for model prediction
# This step handles scaling and normalization for Xception
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)


# In[9]:


# Make predictions
predictions = model.predict(x)
print('Predicted:', decode_predictions(predictions, top=3)[0])
plt.imshow(img)


# In[56]:


# Refactor above steps into reusable function
def predict(image_path):
    """Use Xception to label image"""
    img = image.load_img(image_path, target_size=image_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    predictions = model.predict(x)
    plt.imshow(img)
    print('Predicted:', decode_predictions(predictions, top=1)[0])
    return decode_predictions(predictions, top=1)[0]


# In[61]:


image_path = os.path.join("Images", "pup.jpg")
prediction = predict(image_path)
breed = prediction[0][1]


# In[62]:


import pandas as pd
animal_outcome = pd.read_csv('../Animals.csv')
animal_outcome.head()


# In[65]:


print(breed)
animal_outcome.loc[animal_outcome['Breed'].str.lower().str.contains(breed, regex=True)]
# animal_outcome.loc[animal_outcome['Breed'].str.contains('border collie', regex=False)


# In[83]:


from fuzzywuzzy import process
from fuzzywuzzy.fuzz import partial_ratio
test_process = process.extract(breed.replace('_', ' '), animal_outcome.Breed.to_dict(), scorer=partial_ratio, limit=100000000)


# In[90]:


r = process.extractBests(breed.replace('_', ' '), animal_outcome.Breed.to_dict(), scorer=partial_ratio, score_cutoff=70, limit=1000000000)
animal_analysis = animal_outcome.loc[map(lambda x: x[-1], r)]

