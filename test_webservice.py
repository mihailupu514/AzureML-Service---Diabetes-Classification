import json
from azureml.core import Workspace
from azureml.core.webservice import AciWebservice
from azureml.core.model import InferenceConfig

# Load the workspace from the saved config file
ws = Workspace.from_config()
print('Ready to work with', ws.name)

for webservice_name in ws.webservices:
    webservice = ws.webservices[webservice_name]
    print(webservice.name)

service = ws.webservices['diabetes-service']

x_new = [[2,180,74,24,21,23.9091702,1.488172308,22]]
print ('Patient: {}'.format(x_new[0]))

# Convert the array to a serializable list in a JSON document
input_json = json.dumps({"data": x_new})

# Call the web service, passing the input data (the web service will also accept the data in binary format)
predictions = service.run(input_data = input_json)

# Get the predicted class - it'll be the first (and only) one.
predicted_classes = json.loads(predictions)
print(predicted_classes[0])

endpoint = service.scoring_uri
print(endpoint)