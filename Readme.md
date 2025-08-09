# Breast Cancer Detection Project

## Overview

This project develops a deep learning model for **breast cancer detection** using a **custom CNN** trained on an ultrasound image dataset. The model classifies images into **four categories** — three known breast cancer types and a new category called **"Unknown"**.
The trained model is deployed using **Streamlit** for an easy-to-use web interface.

---

## Dataset Details

* **Source:** \[Insert dataset source name or link here]
* **Structure:**

  * 4 classes: `Unknown`, `Benign`, `Malignant`, `Normal`
  * Images organized in separate folders per class
* **Size:** \[Insert number of images per class and total]
* **Features:** Ultrasound breast images with varied resolution and image quality

---

## Model Details

* **Architecture:** Custom CNN built with TensorFlow/Keras
* **Input Shape:** `(224, 224, 3)`
* **Output Classes:** 4 (softmax activation)
* **Loss Function:** `categorical_crossentropy`
* **Optimizer:** `Adam`
* **Metrics:** Accuracy, precision, recall, F1-score

---

## Code Structure

### 1. **Data Preprocessing**

```python
import os
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Generate data paths with labels
data_dir = 'C:/Users/Joshua/Desktop/Dataset_BUSI_with_GT'
filepaths, labels = [], []

folds = os.listdir(data_dir)
for fold in folds:
    foldpath = os.path.join(data_dir, fold)
    filelist = os.listdir(foldpath)
    for file in filelist:
        fpath = os.path.join(foldpath, file)
        filepaths.append(fpath)
        labels.append(fold)

# Combine into DataFrame
df = pd.DataFrame({'filepaths': filepaths, 'labels': labels})

# Train-test split
y = df['labels']
train_df, test_df = train_test_split(df, train_size=0.8, shuffle=True, random_state=123, stratify=y)
```

---

### 2. **Model Training**

```python
from tensorflow.keras import models, layers

# Build the CNN
model = models.Sequential()
model.add(layers.Input(shape=(224, 224, 3)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(rate=0.45, seed=123))
model.add(layers.Dense(4, activation='softmax'))

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

---

### 3. **Model Evaluation**

```python
# Make predictions
preds = model.predict(test_gen)
y_pred = np.argmax(preds, axis=1)

from sklearn.metrics import confusion_matrix
import itertools

# Confusion Matrix
g_dict = test_gen.class_indices
classes = list(g_dict.keys())
cm = confusion_matrix(test_gen.classes, y_pred)

plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j], horizontalalignment='center',
             color='white' if cm[i, j] > thresh else 'black')

plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()
```

---

### 4. **Deployment (Streamlit App)**

```python
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np

st.title("Breast Cancer Classification")

# Load trained model
@st.cache(allow_output_mutation=True)
def load_trained_model():
    model = load_model('testmodel2.h5')
    return model

model = load_trained_model()

# Preprocess image
def preprocess_image(image):
    image = image.resize((224, 224))
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    img = preprocess_image(image)
    prediction = model.predict(img)

    labels = ['Unknown', 'Benign', 'Malignant', 'Normal']
    predicted_class = labels[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.write(f"Predicted Class: {predicted_class} with {confidence * 100:.2f}% confidence.")
```

---

##  Requirements

* Python: 3.11
* Libraries:

  * TensorFlow / Keras
  * NumPy
  * Pandas
  * Matplotlib / Seaborn
  * scikit-learn
  * PIL
  * Streamlit

Install with:

```bash
pip install -r requirements.txt
```

---

##  Usage

1. Clone the repository:

```bash
git clone https://github.com/[your-username]/breast-cancer-detection.git
```

2. Install requirements:

```bash
pip install -r requirements.txt
```

3. Run Streamlit app:

```bash
streamlit run app.py
```
