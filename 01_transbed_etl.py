# Databricks notebook source
# MAGIC %md
# MAGIC # Merchant Category Code
# MAGIC 
# MAGIC Merchant Category Codes (MCC) is a standard ([ISO18245](https://www.iso.org/standard/33365.html)) in the payment industry for classifying card payment terminals. Although useful for operation / authorization purposes, such a rigid taxonomy doesn't fully capture the specificities across merchants that could help us identify customers' shopping behaviors. Would the same MCC used to cover both men's luxury and streetwear brands help us fully capture customer preferences? This is where our solution comes into play, moving from merchant categorization that is no longer based on standards, but based on customer's behaviors (i.e. based on data). We assume that by learning specificities across merchants, we will be able to better capture customer shopping preferences that could be used for advanced customer 360 use cases.

# COMMAND ----------

# MAGIC %md
# MAGIC <img src='https://instabill.com/wp-content/uploads/2017/05/merchant-category-code-instabill-370x230.jpg' width=500>
# MAGIC 
# MAGIC MCC codes, see [list](https://www.citibank.com/tts/solutions/commercial-cards/assets/docs/govt/Merchant-Category-Codes.pdf) from citigroup

# COMMAND ----------

# MAGIC %md
# MAGIC # Graph based approach
# MAGIC Whilst common sense would be to approach a segmentation exercise as a simple clustering model over a set of categorical variables (how often did customer *C* visit shop *S*), there are only a few off-the-shelf techniques that could be used, not to mention an obvious problem of scale (there are currently 10 million payment terminals in the US, hence 10 million of categorical variables). However, when converting data from its original archetype into another (such as graph), we can access a wider range of techniques that often yield unexpected results, surprising even. In this notebook, we will convert our original card transaction data into graph paradigm and access techniques originally designed for... Natural Language Processing!

# COMMAND ----------

# MAGIC %run ./config/configure_notebook

# COMMAND ----------

# MAGIC %md
# MAGIC For the purpose of that exercise, we will use a card transaction dataset (not included in the solution) but only focus on customer identifier, transaction date and merchant narrative that has been previously classified as a brand using our transaction enrichment [solution accelerator](https://databricks.com/blog/2021/05/10/improving-customer-experience-with-transaction-enrichment.html). As reported in our previous solution, it is expected to observe discrepancies in the number of transactions made between large retailers and local shops. Organizations like Walmart or Costco (two main US based retailers) would account for much more transactions than our "little shop" point of sales terminal just across the street. Whatever the approach defined later will have to address that disparity.

# COMMAND ----------

from pyspark.sql import functions as F

transactions_raw = (
  spark
    .read
    .format('delta')
    .load(config['data']['raw'])
    .select(
      F.col('tr_date').alias('date'),
      F.col('cs_reference').alias('customer_id'),
      F.col('tr_merchant').alias('merchant_name'),
      F.col('tr_amount').alias('amount')
    )
)

# COMMAND ----------

from pyspark.sql.window import Window

merchants = (
  transactions_raw
    .select('merchant_name', 'customer_id')
    .distinct()
    .groupBy('merchant_name')
    .count()
    .select(
      F.col('merchant_name').alias('id'),
      F.col('merchant_name').alias('label'),
      F.col('count').alias('unique_customers_total')
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Shopping graph
# MAGIC In this section, we will be building a large network made of millions of customers and thousands of brands with associated card transactions. A naive approach would be to connect every customer with every shop they ever visited. However, this may lead to a highly connected graph. In order to limit the number of connections, we filter our graph using a temporal condition, such as a shop *A* would be connected to a shop *B* only if there exists a same customer *C* who visited both shops within a given timeframe (arbitrarily set to 2 days here). As readers can notice, we are implicitly projecting our bipartite network (customer <> merchant) to a one-partite graph (merchant <> merchant) given the number of unique visitors they both have in common. 

# COMMAND ----------

# MAGIC %md
# MAGIC <img src=https://raw.githubusercontent.com/databricks-industry-solutions/transaction-embedding/main/images/transbed_graph.png width="600px">

# COMMAND ----------

from pyspark.sql.window import Window

days = lambda i: i * 86400 

# our temporal window function
next_x_days = (
  Window
    .partitionBy(F.col('customer_id'))
    .orderBy(F.col('date').cast('timestamp').cast('long'))
    .rangeBetween(0, days(config['model']['days']))
)

# connect brands when visited by a same customer within a given timeframe
transitions = (
  transactions_raw
    .withColumnRenamed('merchant_name', 'label')
    .join(merchants, ['label'])
    .withColumn('next_x_days_visits', F.collect_set(F.col('id')).over(next_x_days))
    .select('customer_id', 'id', 'date', 'next_x_days_visits')
    .withColumn('next_x_days_visits', F.explode('next_x_days_visits'))
    .select(
      F.col('customer_id'),
      F.col('id').alias('src'),
      F.col('next_x_days_visits').alias('dst')
    )
    .filter(F.col('src') != F.col('dst')) # prevent self loops
    .distinct() # unique customers
    .groupBy('src', 'dst')
    .count() # number of unique customers both src and dst have in common
    .withColumnRenamed('count', 'unique_customers_x_days')
    .cache()
)

display(transitions)

# COMMAND ----------

# MAGIC %md
# MAGIC Before attempting to learn those mathematical relationships, we may have to better understand our graph topology, i.e. is there any merchant that would connect everyone with everyone else? A first approach could be to run a simple degree count from the `graphframes` library (part of a databricks ML runtime) to get a sense of how connected our graph is. Below visualization shows that many brands are connected to the entire graph, as expected from our previous work on transaction enrichment.

# COMMAND ----------

from graphframes import *
graph = GraphFrame(merchants, transitions)
display(graph.outDegrees)

# COMMAND ----------

# MAGIC %md
# MAGIC For the purpose of this solution, we will be ignoring our 20% most connected brands. The reasoning is that not much can be learned from a brand if our entire population visited the same merchant. Those thresholds would need to be driven from our data by studying our graph topology.

# COMMAND ----------

q10, q80 = graph.outDegrees.approxQuantile('outDegree', [0.1, 0.8], 0.1)

nodes = (
  graph
    .outDegrees.where((F.col('outDegree') > q10) & (F.col('outDegree') < q80))
    .join(merchants.select('id', 'unique_customers_total'), ['id'])
    .withColumnRenamed('id', 'label')
    .withColumn('id', F.row_number().over(Window.orderBy(F.desc('unique_customers_total'))) - 1)
    .select('id', 'label', 'unique_customers_total')
    .cache()
)

edges = (
  transitions
    .join(nodes.select(F.col('label').alias('src'), F.col('id')), ['src']).drop('src').withColumnRenamed('id', 'src')
    .join(nodes.select(F.col('label').alias('dst'), F.col('id')), ['dst']).drop('dst').withColumnRenamed('id', 'dst')
    .select("src", "dst", "unique_customers_x_days")
)

# COMMAND ----------

edges.write.format('delta').mode('overwrite').save('{}/merchant_edges'.format(home_dir))
nodes.write.format('delta').mode('overwrite').save('{}/merchant_nodes'.format(home_dir))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Shopping trips
# MAGIC Similar to natural language processing techniques where the meaning of a word is defined by its surrounding context, we assume that the definition of merchant can be learned from its customer base and the other brands they like to shop to (the surrounding context). In order to build such a context, we will be generating random walks, simulating customers "walking" from a shop to another, up and down our graph structure as represented below. The aim will be to learn "embeddings", i.e. a mathematical representation of the contextual information carried by the nodes in our network. Two merchants contextually close from one another would be "embedded" into large vectors mathematically close from one another. By extension, 2 customers who exhibit the same shopping behavior may be mathematically close from one another.

# COMMAND ----------

# MAGIC %md
# MAGIC <img src=https://raw.githubusercontent.com/databricks-industry-solutions/transaction-embedding/main/images/transbed_shopping_trips.gif width="600px">

# COMMAND ----------

# MAGIC %md
# MAGIC As reported below, generating random walks is a relatively trivial task against our graph transition matrix. But even with 1,000 distinct merchants, our graph transition matrix will be 1,000 x 1,000 large. The same would not fit in memory with say, 10,000 or 100,000 merchants. From experience, however, even with millions of card payment terminals in the US, there are many regions, known industries, known segments of customers we know are logically different from one another, hence resulting in many of smaller matrices. In this example, we consider our data as a whole and generate random walks through monte carlo simulations by distributing our transition matrix to each of our spark executors.

# COMMAND ----------

edges_df = spark.read.format('delta').load('{}/merchant_edges'.format(home_dir)).toPandas()
nodes_df = spark.read.format('delta').load('{}/merchant_nodes'.format(home_dir)).toPandas()
merchants_dict = dict(zip(nodes_df.id, nodes_df.label))

# COMMAND ----------

adjacency_df = edges_df.pivot(index='src', columns='dst', values='unique_customers_x_days').fillna(0)

# Ensure matrix is nxn
index = nodes_df.id
adjacency_df = adjacency_df.reindex(index=index, columns=index, fill_value=0)

# normalize to get transition state probability
# given a shop s1, what is the probability of a random user u to visit shop s2 next?
adjacency_df = adjacency_df.div(adjacency_df.sum(axis=1), axis=0).fillna(0)
transition_matrix = adjacency_df.to_numpy()
adjacency_df

# COMMAND ----------

# MAGIC %md
# MAGIC Given a state vector *s1*, we select a new state (a merchant) based on its transition probability, resulting in a new state vector *s2*. By repeating this process *X* times, we generate a new random trip of size *X*. In the context of this exercise, we will be generating 10,000 shopping trips originating from each merchant in our database. Although we initially chose our shopping trips dimensions of 5, arbitrarily, we did not detect significant improvements in our model (see next notebooks) with larger shopping trips. We recommend testing different parameters, though, as it strongly depends on the dataset at play.

# COMMAND ----------

from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import ArrayType, StringType, StructType, StructField, IntegerType
import numpy as np
import pandas as pd

m_B = spark.sparkContext.broadcast(transition_matrix)
b_B = spark.sparkContext.broadcast(merchants_dict)

schema = StructType([StructField('walks', ArrayType(StringType()), True)])

@pandas_udf(schema, PandasUDFType.GROUPED_MAP)
def walk_udf(key, df):
  
  # deserialize our transition matrix
  m = m_B.value
  b = b_B.value
  
  # creating our initial merchant state
  i = key[0]
  state_vector = np.zeros(shape=(1, m.shape[0]))
  state_vector[0][i] = 1.0
  
  shopping_trips = []
  
  # for each simulation of size X
  for t in np.arange(0, config['model']['nums']):

    walks = [b[i]]
    for s in np.arange(0, config['model']['size'] - 1):
      
      # generate our distribution for our next move
      pvals = np.dot(state_vector, m)[0]
      
      # sample a point from our probability distribution
      state_id = np.where(np.random.multinomial(1,pvals))[0][0]
      
      # walk towards this shop
      state_vector = np.zeros(shape=(1, m.shape[0]))
      state_vector[0][state_id] = 1.0
      
      # get a "stamp" from this shop and prepare to move to next merchant
      next_hop = b[state_id]
      walks.append(next_hop)
      
    shopping_trips.append([walks])
  return pd.DataFrame(shopping_trips, columns=['walks'])

# COMMAND ----------

from pyspark.sql import functions as F
shopping_trips = spark.read.format('delta').load('{}/merchant_nodes'.format(home_dir)).groupBy('id').apply(walk_udf).cache()
display(shopping_trips)

# COMMAND ----------

shopping_trips.write.format('delta').mode('overwrite').save('{}/shopping_trips'.format(home_dir))

# COMMAND ----------

# MAGIC %md
# MAGIC In this notebook, we have transformed our original card transaction data into a large network of merchants connected based on the number of customers they share in common. We assumed that a merchant category could be learned from their transactional context, just like words could be learned from their surrounding context in the world of NLP. We have introduced the concepts of embedding and generated the required dataset through simple matrix computations. In the next notebook, more focused on ML, we will try to confirm our hypothesis and use those findings to enrich our customers' views.
