#!/usr/bin/env python
import tensorflow as tf
import numpy
from PIL import Image

def load_and_preprocess(image_path, image_size=224):
    """Load the image specified at image_path and preprocess the image in the 
    same manner as the Smart Skin Cancer Detection app. The image is resized 
    to the classifier's native input size of 224 px by 224 px, warping rather 
    than cropping, using nearest-neighbor rescaled without antialiasing. Note
    that the implementation of nearest-neighbor interpolation is consistent 
    between PIL and Android. The image is then scaled to the range (-1, 1).
    
    Args:
        image_path (str): The file path to the image to load."""
    image = Image.open(image_path).resize((image_size, image_size), resample=Image.NEAREST)
    image = numpy.array(image, dtype=numpy.uint8)
    image = numpy.require(image, dtype=numpy.float32)
    image = (image-128)/128
    image = image[numpy.newaxis,:]
    return image

def load_and_preprocess_cropped(image_path, image_size=224):
    """Load the image specified at image_path and preprocess the image. The 
    image is CENTER-CROPPED to a square, then resized to the classifier's 
    native input size of 224 px by 224 using nearest-neighbor rescaled without 
    antialiasing. Note that the implementation of nearest-neighbor 
    interpolation is consistent between PIL and Android. The image is then 
    scaled to the range (-1, 1).

    ..note: This method makes a different assumption about how a user might 
    crop the input image than the default `load_and_preprocess` function.
    
    Args:
        image_path (str): The file path to the image to load."""
    image = Image.open(image_path)
    width, height = image.size
    if height > width:
        image = image.crop((0,int(height/2-width/2),width,int(height/2+width/2)))
    elif width > height:
        image = image.crop((int(width/2-height/2),0,int(width/2+height/2),height))
    image = image.resize((image_size, image_size), resample=Image.NEAREST)
    image = numpy.array(image, dtype=numpy.uint8)
    image = numpy.require(image, dtype=numpy.float32)
    image = (image-128)/128
    image = image[numpy.newaxis,:]
    return image

class TFLiteSSCDClassifier(object):
    '''
    Python API for tensorflow lite version of Smart Skin Cancer Detection.

    This model is incompatible with PyTorch and the explainable AI tools used 
    elsewhere in this package. The primary use of this model is to evaluate the
    quality of the reproduction of the PyTorch implementation of Smart Skin 
    Cancer Detection.

    Args:
        model_path (str): The file path to the .tflite file.
    '''
    def __init__(self, model_path, image_size=224):
        self.labels = ['none', 'melanoma', 'nevus', 'seborrheic keratosis']
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self._input_details = self.interpreter.get_input_details()
        self._output_details = self.interpreter.get_output_details()
        self.image_size=image_size

    def run_from_filepath(self, image_path):
        """Run the model on the specified image path.

        Args:
            image_path (str): The file path to the image to analyze."""
        # load image as uint8 and resize to (self.image_size, self.image_size), warping image if necessary
        # for reproducing app exactly, nearest-neighbor interpolation for resizing is MANDATORY
        image = load_and_preprocess(image_path, self.image_size)
        return self.run(image)

    def run(self, image):
        """Run the model on the specified image array.
        
        Args:
            image (numpy.ndarray): An array of shape 
                (self.image_size, self.image_size, 3) representing the image to 
                evaluate. The image should be in the range (0, 255)."""
        self.interpreter.set_tensor(self._input_details[0]["index"], image)
        self.interpreter.invoke()
        tflite_interpreter_output = self.interpreter.get_tensor(self._output_details[0]["index"])
        probabilities = numpy.array(tflite_interpreter_output[0])
        output = numpy.require(probabilities, dtype=numpy.float32)/256
        return output/output.sum()
