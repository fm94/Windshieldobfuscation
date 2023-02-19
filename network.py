import tensorflow as tf

# class SimpleUNet:
#     # this is a really bad implementation but it does the job
#     def __init__(self, lr = 0.001, in_channels = 3, out_channels = 1, loss = 'mse', metrics = ['mse']):
#         self.lr = lr
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.get_opt()
#         self.encoder()
#         self.decoder()
#         self.autoencoder = self.get_model()
#         self.autoencoder.compile(
#                     loss=loss, 
#                     optimizer=self.opt, 
#                     metrics=metrics)
#         return self.autoencoder
        
#     def encoder(self):
#         self.input_layer = tf.keras.Input(shape=(None, None, self.in_channels))
#         self.x1 = tf.keras.layers.Conv2D(16, (5, 5), activation='relu', padding='same')(self.input_layer)
#         self.x2 = tf.keras.layers.Conv2D(16, (5, 5), activation='relu', padding='same')(self.x1)
#         self.x3 = tf.keras.layers.MaxPool2D(padding='same')(self.x2)
#         self.x4 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(self.x3)
#         self.x5 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(self.x4)
#         self.x6 = tf.keras.layers.MaxPool2D(padding='same')(self.x5)
#         self.encoded = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(self.x6)

#     def decoder(self, encoded):
#         self.x7 = tf.keras.layers.UpSampling2D()(encoded)
#         self.x8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(self.x7)
#         self.x9 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(self.x8)
#         self.x10 = tf.keras.layers.Add()([self.x5, self.x9])
#         self.x11 = tf.keras.layers.UpSampling2D()(self.x10)
#         self.x12 = tf.keras.layers.Conv2D(16, (5, 5), activation='relu', padding='same')(self.x11)
#         self.x13 = tf.keras.layers.Conv2D(16, (5, 5), activation='relu', padding='same')(self.x12)
#         self.x14 = tf.keras.layers.Add()([self.x2, self.x13])
#         self.decoded = tf.keras.layers.Conv2D(self.out_channels, (3, 3), padding='same',activation='relu')(self.x14)

#     def get_model(self):
#         autoencoder = tf.keras.Model(self.input_layer, self.decoded)
#         return autoencoder
    
#     def get_opt(self):
#         self.opt = tf.keras.optimizers.Adam(learning_rate=self.lr)
        
        
def SimpleUNet(lr = 0.001, in_channels = 3, out_channels = 1, loss = 'mse', metrics = ['mse']):
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    input_img = tf.keras.Input(shape=(None, None, in_channels))
        
    # encoder architecture
    x1 = tf.keras.layers.Conv2D(16, (5, 5), activation='relu', padding='same')(input_img)
    x2 = tf.keras.layers.Conv2D(16, (5, 5), activation='relu', padding='same')(x1)
    x3 = tf.keras.layers.MaxPool2D(padding='same')(x2)
    x4 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x3)
    x5 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x4)
    x6 = tf.keras.layers.MaxPool2D(padding='same')(x5)
    encoded = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x6)

    # decoder architecture
    x7 = tf.keras.layers.UpSampling2D()(encoded)
    x8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x7)
    x9 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x8)
    x10 = tf.keras.layers.Add()([x5, x9])
    x11 = tf.keras.layers.UpSampling2D()(x10)
    x12 = tf.keras.layers.Conv2D(16, (5, 5), activation='relu', padding='same')(x11)
    x13 = tf.keras.layers.Conv2D(16, (5, 5), activation='relu', padding='same')(x12)
    x14 = tf.keras.layers.Add()([x2, x13])
    decoded = tf.keras.layers.Conv2D(out_channels, (3, 3), padding='same',activation='relu')(x14)
    autoencoder = tf.keras.Model(input_img, decoded)
    autoencoder.compile(loss=loss, optimizer=opt, metrics=metrics)
    return autoencoder