import os
import azureml.core
from azureml.core import Workspace, Experiment
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core.runconfig import RunConfiguration
from azureml.core.conda_dependencies import CondaDependencies
from azureml.pipeline.core import PipelineData, Pipeline
from azureml.pipeline.steps import PythonScriptStep, EstimatorStep
from azureml.train.estimator import Estimator
from azureml.widgets import RunDetails


print("Ready to use Azure ML", azureml.core.VERSION)

# Load the workspace from the saved config file
ws = Workspace.from_config()
print('Ready to work with', ws.name)

# Create a folder for the experiment files
experiment_name = 'diabetes_pipeline'
experiment_folder = './' + experiment_name
os.makedirs(experiment_folder, exist_ok=True)

#Fetch CPU cluster for computations
cpu_cluster = ComputeTarget(workspace=ws, name='cpu-compute')

#Create python environment
# Create a new runconfig object
pipeline_run_config = RunConfiguration()

# Use the compute you created above. 
pipeline_run_config.target = cpu_cluster

# Enable Docker
pipeline_run_config.environment.docker.enabled = True

# Specify CondaDependencies obj, add necessary packages
pipeline_run_config.environment.python.user_managed_dependencies = False
pipeline_run_config.environment.python.conda_dependencies = CondaDependencies.create(
    conda_packages=['pandas','scikit-learn'], 
    pip_packages=['azureml-sdk','argparse'])

print ("Run configuration created.")

#Define pipeline
# Get the training dataset
diabetes_ds = ws.datasets.get("diabetes_dataset")

# Create a PipelineData (Data Reference) for the model folder
model_folder = PipelineData("model_folder", datastore=ws.get_default_datastore())

estimator = Estimator(source_directory=experiment_folder,
                        compute_target = cpu_cluster,
                        environment_definition=pipeline_run_config.environment,
                        entry_script='train_diabetes.py')

#Step 1
train_step = EstimatorStep(name = "Train Model",
                           estimator=estimator, 
                           estimator_entry_script_arguments=['--regularization', 0.1,
                                                             '--output_folder', model_folder],
                           inputs=[diabetes_ds.as_named_input('diabetes')],
                           outputs=[model_folder],
                           compute_target = cpu_cluster,
                           allow_reuse = False)

# Step 2, run the model registration script
register_step = PythonScriptStep(name = "Register Model",
                                source_directory = experiment_folder,
                                script_name = "register_diabetes.py",
                                arguments = ['--model_folder', model_folder],
                                inputs=[model_folder],
                                compute_target = cpu_cluster,
                                runconfig = pipeline_run_config,
                                allow_reuse = False)

print("Pipeline steps defined")

#Run pipeline

# Construct the pipeline
pipeline_steps = [train_step, register_step]
pipeline = Pipeline(workspace = ws, steps=pipeline_steps)
print("Pipeline is built.")

# Create an experiment and run the pipeline
experiment = Experiment(workspace = ws, name = experiment_name)
pipeline_run = experiment.submit(pipeline, regenerate_outputs=True)
print("Pipeline submitted for execution.")