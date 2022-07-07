# Databricks notebook source
import warnings
warnings.filterwarnings("ignore")

# COMMAND ----------

import re
from pathlib import Path

# We ensure that all objects created in that notebooks will be registered in a user specific database. 
useremail = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
username = useremail.split('@')[0]

# Please replace this cell should you want to store data somewhere else.
database_name = '{}_transbed'.format(re.sub('\W', '_', username))
_ = sql("CREATE DATABASE IF NOT EXISTS {}".format(database_name))

# Similar to database, we will store actual content on a given path
home_directory = '/FileStore/{}/transbed'.format(username)
dbutils.fs.mkdirs(home_directory)

# Where we might stored temporary data on local disk
temp_directory = "/tmp/{}/transbed".format(username)
Path(temp_directory).mkdir(parents=True, exist_ok=True)

# COMMAND ----------

import re

config = {
  'num_executors'             :  '8',
  'model_name'                :  'transbed_{}'.format(re.sub('\.', '_', username)),
  'transactions_raw'          :  '/mnt/industry-gtm/fsi/datasets/card_transactions',
  'merchant_edges'            :  '{}/merchant_edges'.format(home_directory),
  'merchant_nodes'            :  '{}/merchant_nodes'.format(home_directory),
  'shopping_trips'            :  '{}/shopping_trips'.format(home_directory),
  'merchant_vectors'          :  '{}/merchant_vectors'.format(home_directory),
  'shopping_trip_size'        :  '5',
  'shopping_trip_days'        :  '2',
  'shopping_trip_number'      :  '1000',
}

# COMMAND ----------

import mlflow
experiment_name = f"/Users/{useremail}/transbed"
mlflow.set_experiment(experiment_name) 

# COMMAND ----------

import pandas as pd
 
# as-is, we simply retrieve dictionary key, but the reason we create a function
# is that user would be able to replace dictionary to application property file
# without impacting notebook code
def getParam(s):
  return config[s]
 
# passing configuration to scala
spark.createDataFrame(pd.DataFrame(config, index=[0])).createOrReplaceTempView('esg_config')

# COMMAND ----------

# MAGIC %scala
# MAGIC val cdf = spark.read.table("esg_config")
# MAGIC val row = cdf.head()
# MAGIC val config = cdf.schema.map(f => (f.name, row.getAs[String](f.name))).toMap
# MAGIC def getParam(s: String) = config(s)

# COMMAND ----------

def tear_down():
  import shutil
  try:
    shutil.rmtree(temp_directory)
  except:
    pass
  dbutils.fs.rm(home_directory, True)
  _ = sql("DROP DATABASE IF EXISTS {} CASCADE".format(database_name))

# COMMAND ----------

from pyspark.sql import functions as F

transactions_raw = (
  spark
    .read
    .format('delta')
    .load(getParam('transactions_raw'))
    .select(
      F.col('tr_date').alias('date'),
      F.col('cs_reference').alias('customer_id'),
      F.col('tr_merchant').alias('merchant_name'),
      F.col('tr_amount').alias('amount')
    )
)

transactions_raw.createOrReplaceTempView("transactions")
