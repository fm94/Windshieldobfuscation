import os
import tensorflow as tf
import numpy as np
from glob import glob

from network import SimpleUNet
from dataset import DataLoader

train_mask_dir = "data/annotations/train/"
val_mask_dir = "data/annotations/test/"

batch_size = 16
epochs = 30
target_shape = (64, 64)

train_images = glob(os.path.join(train_mask_dir, "*"))
val_images = glob(os.path.join(val_mask_dir, "*"))

# Callbacks
early_stop_cb = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=10,
    verbose=1,
    mode="auto",
    baseline=None,
    restore_best_weights=True,
)
model_ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
    "my_weights.h5",
    monitor="val_loss",
    verbose=0,
    save_best_only=True,
    save_weights_only=False,
    mode="auto",
    save_freq="epoch",
)
reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.1,
    patience=4,
    verbose=1,
    mode="auto",
    min_delta=0.0001,
    cooldown=0,
    min_lr=10e-8,
)

autoencoder = SimpleUNet()
autoencoder.summary()

train_dataset = DataLoader(train_mask_dir, target_shape=target_shape)
train_generator = train_dataset.data_generator(batch_size)

val_dataset = DataLoader(val_mask_dir, target_shape=target_shape)
val_generator = val_dataset.data_generator(batch_size)

train_steps = len(list(train_dataset.files)) // batch_size + 1
val_steps = len(list(val_dataset.files)) // batch_size + 1

# load pretrained model - to continue training
#autoencoder.load_weights('my_weights.h5')

# Train
history = autoencoder.fit(train_generator,
                            validation_data=val_generator,
                            batch_size=batch_size, 
                            steps_per_epoch=train_steps,
                            validation_steps=val_steps,
                            epochs=epochs, 
                            callbacks=[
                                early_stop_cb, 
                                model_ckpt_cb, 
                                reduce_lr_cb]
                            )

# history can be saved later on...