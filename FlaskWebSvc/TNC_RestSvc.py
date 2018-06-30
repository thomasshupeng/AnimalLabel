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



CNTK_MODEL_FILE = 'TNC_ResNet18_ImageNet_CNTK.model'
tl_model_file = os.path.join(os.getcwd(), "Model", CNTK_MODEL_FILE)

# define base model location and characteristics
_image_height = 512
_image_width = 682
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


project_name_to_id = {'TNC': '11111111',
                      '云南老君山': '22222222',
                      '北大生命科学院': '33333333'}

# 'tncapi/v1.0/Prediction/cntk21'
SERVICE_NAME = 'tncapi'
API_VERSION = 'v1.0'
END_POINT_NAME = 'Prediction'
MODEL_NAME = 'cntk21'

app = Flask(__name__)
app.config["DEBUG"] = True
# 1.	PredictImageUrl
# https://southcentralus.api.cognitive.microsoft.com/customvision/v1.1/Prediction/{projectId}/url[?iterationId][&application]
endpoint = "/" + SERVICE_NAME + "/" + API_VERSION + "/" + END_POINT_NAME + "/" + project_name_to_id['TNC'] + "/url"
print("end point1 = ", endpoint)

@app.route(endpoint, methods=['POST'])
def get_prediction_img_url():
    print("=== Arguments ===")
    iterationId = request.args.get('iterationId')
    print("iterationId = {!s}".format(iterationId))
    application = request.args.get('application')
    print("application = {!s}".format(application))
    print("=== Headers ===")
    content_type = request.headers.get('Content-Type')
    print("Content-Type = {!s}".format(content_type))
    prediction_key = request.headers.get('Prediction-Key')
    print("Prediction-Key = {!s}".format(prediction_key))
    print("=== Body ===")
    img_url = request.json.get('Url')
    print("Url = {!s}".format(img_url))

    res_prediction_img_url = {
        "Id": "string",
        "Project": project_name_to_id['TNC'],
        "Iteration": iterationId,
        "Created": "string",
        "Predictions": [
            {
                "TagId": "99",
                "Tag": "Monkey",
                "Probability": 0.9999
            }
        ]
    }

    return jsonify(res_prediction_img_url)


@app.errorhandler(404)
def page_not_found(e):
    return "<h1>404</h1><p>The resource could not be found.</p>", 404


app.run()

