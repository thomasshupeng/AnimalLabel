import os

import numpy as np
from flask import Flask, jsonify, request
from PIL import Image

import cntk.io.transforms as xforms
from cntk import Constant, Trainer, load_model, placeholder
from cntk.device import gpu, try_set_default_device
from cntk.io import ImageDeserializer, MinibatchSource, StreamDef, StreamDefs
from cntk.layers import Dense
from cntk.learners import (learning_parameter_schedule, momentum_schedule,
                           momentum_sgd)
from cntk.logging import ProgressPrinter, log_number_of_parameters
from cntk.logging.graph import find_by_name, get_node_outputs
from cntk.losses import cross_entropy_with_softmax
from cntk.metrics import classification_error
from cntk.ops import combine, softmax
from cntk.ops.functions import CloneMethod

'''
1.	PredictImageUrl
https://southcentralus.dev.cognitive.microsoft.com/docs/services/57982f59b5964e36841e22dfbfe78fc1/operations/5a3044f608fa5e06b890f163
                
This one is predicting image by a given a url, which is very similar to what you have with some changes:
a.	Instead pass pic URL as url parameter, it passes URL in http body, which doesn’t need any encoding.  
b.	We can make projectid, iterationid, applicationid all as optional, those id can be useful for different projects(云南老君山，北大生命科学院， etc.)

2.	PredictImage
https://southcentralus.dev.cognitive.microsoft.com/docs/services/57982f59b5964e36841e22dfbfe78fc1/operations/5a3044f608fa5e06b890f164

'''
app = Flask(__name__)

# 'tncai/v1.0/Prediction/cntk21'
SERVICE_NAME = 'tncai'
API_VERSION = 'v1.0'
END_POINT_NAME = 'Prediction'
MODEL_NAME = 'cntk21'


CNTK_MODEL_FILE = 'TNC_ResNet18_ImageNet_CNTK.model'
tl_model_file = os.path.join(os.getcwd(), "Model", CNTK_MODEL_FILE)
# define base model location and characteristics
_image_height = 682
_image_width = 512
_num_channels = 3

# Evaluates a single image using the provided model
def eval_single_image(loaded_model, image_path, image_width, image_height):
    # load and format image (resize, RGB -> BGR, CHW -> HWC)
    img = Image.open(image_path)
    if image_path.endswith("png"):
        temp = Image.new("RGB", img.size, (255, 255, 255))
        temp.paste(img, img)
        img = temp
    resized = img.resize((image_width, image_height), Image.ANTIALIAS)
    bgr_image = np.asarray(resized, dtype=np.float32)[..., [2, 1, 0]]
    hwc_format = np.ascontiguousarray(np.rollaxis(bgr_image, 2))

    ## Alternatively: if you want to use opencv-python
    # cv_img = cv2.imread(image_path)
    # resized = cv2.resize(cv_img, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
    # bgr_image = np.asarray(resized, dtype=np.float32)
    # hwc_format = np.ascontiguousarray(np.rollaxis(bgr_image, 2))

    # compute model output
    arguments = {loaded_model.arguments[0]: [hwc_format]}
    output = loaded_model.eval(arguments)

    # return softmax probabilities
    sm = softmax(output[0])
    return sm.eval()



@app.route("/" + SERVICE_NAME + "/" + API_VERSION + "/" + END_POINT_NAME)
def get_prediction():
  return "Hello, World!"

app.run()
def main():
    
    app.run()
    return

if __name__ == '__main__':
    main()
