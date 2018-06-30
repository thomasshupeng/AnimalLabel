import http.client, urllib.request, urllib.parse, urllib.error, base64
import json

headers = {
    # Request headers
    'Content-Type': 'application/json',
    'Prediction-key': '{subscription key}',
}

params = urllib.parse.urlencode({
    # Request parameters
    'iterationId': 'iteration#1',
    'application': 'Test_PredictImgUrl.py',
})

body = {"Url": "https://theartmad.com/wp-content/uploads/2015/02/Cute-Baby-Monkeys-3.jpg"}
json_body = json.dumps(body)
try:
    # conn = http.client.HTTPSConnection('southcentralus.api.cognitive.microsoft.com')
    # conn.request("POST", "/customvision/v1.1/Prediction/{projectId}/url?%s" % params, "{body}", headers)
    conn = http.client.HTTPConnection('127.0.0.1', port=5000)
    conn.request("POST", "/tncapi/v1.0/Prediction/11111111/url?{!s}".format(params), json_body, headers)
    response = conn.getresponse()
    data = response.read()
    print(data)
    json_response = json.loads(data)
    predictions = json_response.get('Predictions')
    if len(predictions)>=1:
        for pred in predictions:
            print("I'm {:f} sure this is a {!s}!".format(pred.get('Probability'), pred.get('Tag')))
    conn.close()
except Exception as e:
    print("[Errno {0}] {1}".format(e.errno, e.strerror))
