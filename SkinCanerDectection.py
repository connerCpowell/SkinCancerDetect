# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%

import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
 
from glob import glob
from PIL import Image
from sklearn.model_selection import train_test_split
 
import tensorflow as tf
from tensorflow import keras
from keras import layers
from functools import partial
 
AUTO = tf.data.experimental.AUTOTUNE
import warnings
warnings.filterwarnings('ignore')


# %%

images = glob('data/*/*.jpg')
len(images)


# %%
images = [path.replace('\\', '/') for path in images]
df = pd.DataFrame({'filepath': images})
df['label'] = df['filepath'].str.split('/', expand=True)[1]
df.head()

# %%
df['label'] = df['label'].str.replace('ISIC-', '')
#df.head(50)

random_sample = df.sample(n=25)
print(random_sample)

# %%

df['label_bin'] = np.where(df['label'].values == 'malignant', 1, 0)
df.head()


# %%

x = df['label'].value_counts()
plt.pie(x.values,
        labels=x.index,
        autopct='%1.1f%%')
plt.show()


# %%

for cat in df['label'].unique():
    temp = df[df['label'] == cat]
 
    index_list = temp.index
    fig, ax = plt.subplots(1, 4, figsize=(15, 5))
    fig.suptitle(f'Images for {cat} category . . . .', fontsize=20)
    for i in range(4):
        index = np.random.randint(0, len(index_list))
        index = index_list[index]
        data = df.iloc[index]
 
        image_path = data[0]
 
        img = np.array(Image.open(image_path))
        ax[i].imshow(img)
plt.tight_layout()
plt.show()


# %%

features = df['filepath']
target = df['label']
 
X_train, X_val, Y_train, Y_val = train_test_split(features, target,
                                      test_size=0.15,
                                      random_state=10)
 
#X_train.shape, X_val.shape, Y_train.shape , Y_val.shape
#Y_val.head()


# %%
tdy = list(Y_train)
tdx = list(X_train)

len(set(tdx) - set(tdy))


# %%

def decode_image(filepath, label):
 
    img = tf.io.read_file(filepath)
    img = tf.image.decode_jpeg(img)
    img = tf.image.resize(img, [224, 224])
    img = tf.cast(img, tf.float32) / 255.0

    
    print('label:', label)
    if label == 'benign':
        Label = 0
    else:
        Label = 1
 
    return img, Label
    return img




# %%
# def decode_image(filepath, label=None):
 
#     img = keras.utils.load_img(filepath, target_size=(224, 224))
#     x = keras.utils.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     x = preprocess_input(x)


#     print(x) 
#     return x

# %%

train_ds = (
    tf.data.Dataset
    .from_tensor_slices((X_train, Y_train))
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(32)
    .prefetch(AUTO)
)
 
val_ds = (
    tf.data.Dataset
    .from_tensor_slices((X_val, Y_val))
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(32)
    .prefetch(AUTO)
)


print(train_ds)

# %%


from tensorflow.keras.applications.efficientnet import EfficientNetB7
 
pre_trained_model = EfficientNetB7(
    input_shape=(224, 224, 3),
    weights='imagenet',
    include_top=False
)
 
for layer in pre_trained_model.layers:
    layer.trainable = False


# %%

from tensorflow.keras import Model
 
inputs = layers.Input(shape=(224, 224, 3))
x = layers.Flatten()(inputs)
 
x = layers.Dense(256, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.3)(x)
x = layers.BatchNormalization()(x)
outputs = layers.Dense(1, activation='sigmoid')(x)
 
model = Model(inputs, outputs)


# %%

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['AUC']
)


# %%

history = model.fit(
                    train_ds,
                    validation_data=val_ds,
                    epochs=5,
                    verbose=1)


# %%

hist_df = pd.DataFrame(history.history)
hist_df.head()


# %%

hist_df['loss'].plot()
hist_df['val_loss'].plot()
plt.title('Loss v/s Validation Loss')
plt.legend()
plt.show()


# %%

hist_df['auc'].plot()
hist_df['val_auc'].plot()
plt.title('AUC v/s Validation AUC')
plt.legend()
plt.show()


# %%
