import os
import azureml.core
from azureml.core import Workspace
from azureml.core import Model
from azureml.core.webservice import AciWebservice
from azureml.core.model import InferenceConfig
from azureml.core.conda_dependencies import CondaDependencies 


print("Ready to use Azure ML", azureml.core.VERSION)

# Load the workspace from the saved config file
ws = Workspace.from_config()
print('Ready to work with', ws.name)

for model in Model.list(ws):
    print(model.name, 'version:', model.version)
    for tag_name in model.tags:
        tag = model.tags[tag_name]
        print ('\t',tag_name, ':', tag)
    for prop_name in model.properties:
        prop = model.properties[prop_name]
        print ('\t',prop_name, ':', prop)
    print('\n')

#get the latest model
model = ws.models['diabetes']
print(model.name, 'version', model.version)

#Create service folder
folder_name = 'diabetes_service'
# Create a folder for the web service files
experiment_folder = './' + folder_name
os.makedirs(folder_name, exist_ok=True)
print(folder_name, 'folder created.')

#Create a container config yml file

# Add the dependencies for our model (AzureML defaults is already included)
myenv = CondaDependencies()
myenv.add_pip_package("scikit-learn")
# myenv.add_pip_package("azureml-sdk[automl]") # Required for AutoML models

# Save the environment config as a .yml file
env_file = folder_name + "/diabetes_env.yml"
with open(env_file,"w") as f:
    f.write(myenv.serialize_to_string())
print("Saved dependency info in", env_file)

# Print the .yml file
with open(env_file,"r") as f:
    print(f.read())

#Deploy web service

# Configure the scoring environment
inference_config = InferenceConfig(runtime= "python",
                                   source_directory = folder_name,
                                   entry_script="score_diabetes.py",
                                   conda_file="diabetes_env.yml")

deployment_config = AciWebservice.deploy_configuration(cpu_cores = 1, memory_gb = 1)

service_name = "diabetes-service-v2"

service = Model.deploy(ws, service_name, [model], inference_config, deployment_config)

service.wait_for_deployment(True)
print(service.state)