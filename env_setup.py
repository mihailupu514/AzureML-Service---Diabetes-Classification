#!pip install --upgrade azureml-sdk[notebooks]
import azureml.core
from azureml.core import Workspace
from azureml.core import ComputeTarget, Datastore, Dataset

print("Ready to use Azure ML", azureml.core.VERSION)
ws = Workspace.from_config()
print(ws.name, "loaded")

#View resources in the workspace 
print("Compute Targets:")
for compute_name in ws.compute_targets:
    compute = ws.compute_targets[compute_name]
    print("\t", compute.name, ":", compute.type)
    
print("Datastores:")
for datastore in ws.datastores:
    print("\t", datastore)
    
print("Datasets:")
for dataset in ws.datasets:
    print("\t", dataset)

print("Web Services:")
for webservice_name in ws.webservices:
    webservice = ws.webservices[webservice_name]
    print("\t", webservice.name)