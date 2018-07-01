"""
File Name: TNC_RestSvc.py
v 1.0

This program provides the RESTful API for TNC project

6/30/2018
Shu Peng
"""
from flask import Flask, jsonify, request
import os
import requests
import datetime
import shutil
import TNC_ModelLoader

'''
1.	PredictImageUrl
https://southcentralus.dev.cognitive.microsoft.com/docs/services/57982f59b5964e36841e22dfbfe78fc1/operations/5a3044f608fa5e06b890f163
                
This one is predicting image by a given a url, which is very similar to what you have with some changes:
a.	Instead pass pic URL as url parameter, it passes URL in http body, which doesn’t need any encoding.  
b.	We can make projectid, iterationid, applicationid all as optional, those id can be useful for different 
projects(云南老君山，北大生命科学院， etc.)

2.	PredictImage
https://southcentralus.dev.cognitive.microsoft.com/docs/services/57982f59b5964e36841e22dfbfe78fc1/operations/5a3044f608fa5e06b890f164

'''

project_name_to_id = {'TNC': '11111111',
                      '云南老君山': '22222222',
                      '北大生命科学院': '33333333'}

SERVICE_NAME = 'tncapi'
API_VERSION = 'v1.0'
END_POINT_NAME = 'Prediction'
MODEL_NAME = '21CResNet18'

# Clean up temp image folder
temp_folder = os.path.join(os.getcwd(), 'temp')
if os.path.exists(temp_folder):
    shutil.rmtree(temp_folder)
    os.makedirs(temp_folder)

loader = TNC_ModelLoader.ModelLoader()
model = loader.get_model(MODEL_NAME)

app = Flask(__name__)
app.config["DEBUG"] = True
# 1.	PredictImageUrl
# https://southcentralus.api.cognitive.microsoft.com/customvision/v1.1/Prediction/{projectId}/url[?iterationId][&application]
endpoint = "/" + SERVICE_NAME + "/" + API_VERSION + "/" + END_POINT_NAME + "/" + project_name_to_id['TNC'] + "/url"
print("end point1 = ", endpoint)


@app.route(endpoint, methods=['POST'])
def get_prediction_img_url():
    print("=== Arguments ===")
    iteration_id = request.args.get('iterationId')
    print("iteration_id = {!s}".format(iteration_id))
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

    # Download image file
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)
    filename = img_url[img_url.rfind('/')+1:]
    img_path_file = os.path.join(temp_folder, filename)

    r = requests.get(img_url, allow_redirects=True, verify=False, auth=('user', 'pass'))
    if r.status_code == requests.codes.ok:
        with open(img_path_file, 'wb') as f:
            f.write(r.content)
    else:
        return jsonify({"Url": img_url, "Code": r.status_code, "Error": r.reason})

    if not os.path.exists(img_path_file):
        res_error = {"Url": img_url, "Error": "Can not open image file."}
        return jsonify(res_error)

    predictions = model.predict(img_path_file)
    os.remove(img_path_file)

    res_prediction_img_url = {
        "Id": "string",
        "Project": project_name_to_id['TNC'],
        "Iteration": iteration_id,
        "Created": datetime.datetime.now().isoformat(),
        "Predictions": predictions}
    return jsonify(res_prediction_img_url)


@app.errorhandler(404)
def page_not_found(e):
    return "<h1>404</h1><p>The resource could not be found.</p>", 404


app.run()
