# Databricks notebook source
# MAGIC %pip install -r requirements.txt

# COMMAND ----------

import yaml
with open('config/application.yaml', 'r') as f:
  config = yaml.safe_load(f)

# COMMAND ----------

import mlflow
username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
mlflow.set_experiment('/Users/{}/transbed'.format(username))

# COMMAND ----------

# Where we might stored temporary data on local disk
temp_directory = "/tmp/{}/transbed".format(username)
Path(temp_directory).mkdir(parents=True, exist_ok=True)
