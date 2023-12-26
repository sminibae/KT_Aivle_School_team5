# imports
import pandas as pd
import numpy as np
import tensorflow as tf

import os
import h5py
import json
import gc
import io

from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization, Dropout, Conv2D, AvgPool2D, Flatten, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.callbacks import Callback


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or '3' to suppress most of the logs
tf.get_logger().setLevel('WARNING')  # Adjust logging level

# Set the GPU memory growth option
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to allocate only the first GPU
        tf.config.set_visible_devices(gpus[0], 'GPU')
        # Enable memory growth to allocate GPU memory on an as-needed basis
        tf.config.experimental.set_memory_growth(gpus[0], True)
        # Set the = activationdesired memory limit (in MB)
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=20480)]  # 20GB
        )
    except RuntimeError as e:
        print(e)

# Load data

# Load one image to get the input shape
with h5py.File('image.h5', 'r') as h5file:
    one_file = h5file['images'][0:1]  # Load the first image

y = np.load('styles.npy', allow_pickle=True)
# Not going to load X yet. because it is too big.
# We are going to load X batch by batch when model.fit.

le = LabelEncoder()
y = le.fit_transform(y)

# Assuming y contains integer labels
y = to_categorical(y, num_classes=7)

# Assuming total number of images
num_images = len(y)  # or len(combined_df)
indices = np.arange(num_images)

# Split indices
indices_train, indices_temp, y_train, y_temp = train_test_split(indices,y, test_size=0.2, random_state=1, stratify=y)
indices_val, indices_test, y_val, y_test = train_test_split(indices_temp, y_temp, test_size=0.5, random_state=1, stratify=y_temp)

np.save('indices_test.npy', np.array(indices_test))

# yield each batch
def data_generator(h5_path, indices, styles, batch_size):
    with h5py.File(h5_path, 'r') as h5file:
        while True:
            np.random.shuffle(indices)
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i + batch_size]
                # Sort the batch_indices to meet HDF5's requirements
                sorted_batch_indices = np.sort(batch_indices)
                batch_images = h5file['images'][sorted_batch_indices]
                batch_styles = styles[sorted_batch_indices]
                yield batch_images, batch_styles


# Create generators
batch_size = 8  # Define your batch size
train_generator = data_generator('image.h5', indices_train, y, batch_size)
val_generator = data_generator('image.h5', indices_val, y, batch_size)
test_generator = data_generator('image.h5', indices_test, y, batch_size)

steps_per_epoch = len(indices_train) // batch_size
validation_steps = len(indices_test) // batch_size



# model
model = Sequential([
    Input(shape=one_file.shape[1:]),
    
    Conv2D(filters=64, kernel_size=(4,4), strides=(1,1), padding='same'),
    BatchNormalization(),
    LeakyReLU(alpha=0.01),
    AvgPool2D(pool_size=(4,4)),
    
    Conv2D(filters=32, kernel_size=(4,4), strides=(1,1), padding='same'),
    BatchNormalization(),
    LeakyReLU(alpha=0.01),
    AvgPool2D(pool_size=(4,4)),
    
    Conv2D(filters=16, kernel_size=(4,4), strides=(1,1), padding='same'),
    BatchNormalization(),
    LeakyReLU(alpha=0.01),
    AvgPool2D(pool_size=(4,4)),
    
    Flatten(),
    Dropout(0.2),
    
    Dense(512),
    BatchNormalization(),
    LeakyReLU(alpha=0.01),
    
    Dense(64),
    BatchNormalization(),
    LeakyReLU(alpha=0.01),
    
    Dense(7),
    Activation('softmax')
])

# Capture the summary output
summary_string = io.StringIO()
model.summary(print_fn=lambda x: summary_string.write(x + '\n'))
summary = summary_string.getvalue()
summary_string.close()

# Write the summary to a file
with open('model_summary.txt', 'w') as file:
    file.write(summary)

# compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss',
                               patience=5,
                               verbose=1,
                               restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.2,
                              patience=5,
                              verbose=1,
                              min_lr=0.001)

model_checkpoint = ModelCheckpoint(filepath="best_model.h5",
                                   monitor='val_loss',
                                   save_best_only=True,
                                   verbose=1)

class GarbageCollectorCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()

    def on_batch_end(self, batch, logs=None):
        gc.collect()
        
gc_callback = GarbageCollectorCallback()

print('start fitting')

history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_generator,
    validation_steps=validation_steps,
    epochs=1000000,  # Set the number of epochs
    verbose=1,
    callbacks=[early_stopping, reduce_lr, model_checkpoint, gc_callback],
)

print('fitting done')

model.save('Conv2D_handmade_model.h5')

# Convert the history.history dict to a JSON file
with open('history.json', 'w') as f:
    json.dump(history.history, f)
