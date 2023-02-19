import tensorflow as tf
import os

class DataLoader:
    def __init__(self, masks_dir, target_shape=(128, 128)):
        masks_dir = self.__fix_glob(masks_dir)
        self.files = tf.data.Dataset.list_files(masks_dir, shuffle=True)
        self.AUTOTUNE = tf.data.experimental.AUTOTUNE
        self.target_shape = target_shape
        
    def __fix_glob(self, dir_path):
        if not dir_path.endswith("*"):
            dir_path = os.path.join(dir_path, "*")
            
        return dir_path
    
    @tf.function
    def parse_images(self, mask_path):
        mask = tf.io.read_file(mask_path)
        mask = tf.io.decode_jpeg(mask, channels=1)
        mask = tf.cast(mask, tf.float32)
        
        image_path = tf.strings.regex_replace(mask_path, "annotations", "images")
        # images are in jpg format, annotations in png
        image_path = tf.strings.regex_replace(image_path, "png", "jpg")
        image = tf.io.read_file(image_path)
        image = tf.io.decode_jpeg(image, channels=3)
        image = tf.cast(image, tf.float32)
        
        mask = tf.image.resize(
            mask, 
            self.target_shape, 
        ) / 255.
        
        image = tf.image.resize(
            image, 
            self.target_shape, 
        ) / 255.
        
        return image, mask
    
    @tf.function
    def data_generator(self, batch_size=32):
        dataset = self.files.map(self.parse_images)
        dataset = dataset.repeat()
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size = self.AUTOTUNE)
        return dataset