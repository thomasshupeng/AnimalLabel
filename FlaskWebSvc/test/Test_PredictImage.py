"""
File Name: Test_PredictImage.py
v 1.0

This program tests the RESTful API (PredictImage) for TNC project

7/1/2018
Shu Peng
"""

import http.client, urllib.request, urllib.parse, urllib.error, base64
import json

headers = {
    # Request headers
    'Content-Type': 'multipart/form-data',
    'Prediction-key': '{subscription key}',
}

params = urllib.parse.urlencode({
    # Request parameters
    'iterationId': 'iteration#2',
    'application': 'Test_PredictImage.py',
})

test_file = "L-LJS17-EBF-0045.JPG"

try:
    with open(test_file, 'rb') as f:
        conn = http.client.HTTPConnection('127.0.0.1', port=5000)
        conn.request("POST", "/tncapi/v1.0/Prediction/11111111/image?{!s}".format(params), body=f.read(), headers=headers)
        response = conn.getresponse()
        data = response.read()
        print(data)
        json_response = json.loads(data)
        predictions = json_response.get('Predictions')
        if len(predictions) >= 1:
            for pred in predictions:
                print("I'm [{!s}] sure this is a {!s}.".format(pred.get('Probability'), pred.get('Tag')))
        conn.close()

except Exception as e:
    print("[Errno {0}] {1}".format(e.errno, e.strerror))
