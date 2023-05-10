#!/usr/bin/env python
import numpy
import tensorflow as tf
from PIL import Image

def load_and_preprocess(image_path, image_size=224, input_dtype=numpy.uint8):
    """Load the image specified at image_path and preprocess the image in the 
    same manner as the Scanoma app. The image is first center-cropped to a 
    square, then rescaled to 300 px by 300 px using bilinear scaling. The image 
    is then resized again to the classifier's native input size of 224 px by 
    224 px. Note that due to differences in the implementation of antialiasing 
    in PIL bilinear interpolation and Android bilinear interpolation, the 
    results are not numerically exact, though extremely close.
    
    Args:
        image_path (str): The file path to the image to load."""
    image = Image.open(image_path)
    width, height = image.size
    if height > width:
        image = image.crop((0,int(height/2-width/2),width,int(height/2+width/2)))
    elif width > height:
        image = image.crop((int(width/2-height/2),0,int(width/2+height/2),height))

    image = image.resize((300,300), resample=Image.BILINEAR)
    image = numpy.array(image.resize((image_size, image_size), 
                                     resample=Image.BILINEAR), 
                        dtype=input_dtype)
    if image.shape[2] == 4: # alpha channel
        image = image[:,:,:3]
    image = image[numpy.newaxis, :,:,:]
    return image

class TFLiteScanomaClassifier(object):
    '''
    Python API for tensorflow lite version of Scanoma.

    This model is incompatible with PyTorch and the explainable AI tools used 
    elsewhere in this package. The primary use of this model is to evaluate the
    quality of the reproduction of the PyTorch implementation of Scanoma.

    Args:
        model_path (str): The file path to the .tflite file.
    '''
    def __init__(self, model_path):
        self.labels = ['benign', 'malignant']
        self.image_size = 224

        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self._input_details = self.interpreter.get_input_details()
        self._output_details = self.interpreter.get_output_details()

    def run_from_filepath(self, image_path):
        """Run the model on the specified image path.

        Args:
            image_path (str): The file path to the image to analyze."""
        input_dtype = self._input_details[0]["dtype"]
        image = load_and_preprocess(image_path, image_size=self.image_size, input_dtype=input_dtype)
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
        return numpy.require(probabilities, dtype=numpy.float32)/255
