# Databricks notebook source
# MAGIC %md
# MAGIC # Transaction embeddings
# MAGIC [Word2Vec](https://arxiv.org/abs/1301.3781) was developed by Tomas Mikolov, et al. at Google in 2013 as a response to make the neural-network-based training of the embedding more efficient and since then has become the de facto standard for developing pre-trained word embedding. As it says on the tin, that model was developed in the context of Natural Language Processing to find similarity of words and algebraic associations like "*man is to king as woman is to ...* ?" (see [paper](http://proceedings.mlr.press/v97/allen19a/allen19a.pdf) from Carl Allen et al.). In the context of card transactions, the aim would be to learn the semantics of a brand given its surrounding context, hence a perfect (albeit surprising) use of such a NLP technique. Could this approach answer questions like "*Starbucks is to Target what Dunkin' Donuts is to ...* ?".

# COMMAND ----------

# MAGIC %run ./config/configure_notebook

# COMMAND ----------

shopping_trips = (
  spark
    .read
    .format('delta')
    .load('{}/shopping_trips'.format(home_dir))
    .repartition(config['model']['exec'])
    .cache()
)

# COMMAND ----------

# MAGIC %md
# MAGIC The main parameters required to tune a `word2vec` model are the window size, vector size and learning rate. However, `word2vec` is rarely used on its own and often associated with a downstream ML model (such as a classification) where an objective function that is known in advance (e.g. improving classification accuracy) could be fed back to our hyperparameter tuning strategy. In our case, we do not have a clear objective function since the merchant taxonomy we want to learn is not known. An approach could be to generate negative / positive sampling and train our own neural network, but we would like to assess the viability of `word2vec` "as-is" before investing time on a more complex ML pipeline. We decided to use a relatively large vector size (255) to capture more granular insights rather than high level categories and apply a small window of 3 given our relatively short shopping trips. 

# COMMAND ----------

import mlflow
from pyspark.ml.feature import Word2Vec

with mlflow.start_run(run_name='shopping_trips') as run:

  mlflow.pyspark.ml.autolog()
  run_id = run.info.run_id
  
  word2Vec = Word2Vec() \
    .setVectorSize(255) \
    .setSeed(42) \
    .setMaxIter(100) \
    .setWindowSize(3) \
    .setMinCount(5) \
    .setInputCol('walks') \
    .setOutputCol('embedding')
  
  # train model
  word2Vec_model = word2Vec.fit(shopping_trips)
  
  # log model
  mlflow.spark.log_model(word2Vec_model, "model")

# COMMAND ----------

# MAGIC %md
# MAGIC As MLFlow captures our experiments in the background, let's register our model candidate. 

# COMMAND ----------

client = mlflow.tracking.MlflowClient()
model_uri = "runs:/{}/model".format(run_id)
result = mlflow.register_model(model_uri, config['model']['name'])
version = result.version

# COMMAND ----------

# MAGIC %md
# MAGIC We can also promote our model to different stages programmatically. Although our models would need to be reviewed in real life scenario, we make it available as a production artifact for our next notebook and programmatically transition previous runs back to Archive.

# COMMAND ----------

client = mlflow.tracking.MlflowClient()
for model in client.search_model_versions("name='{}'".format(config['model']['name'])):
  if model.current_stage == 'Production':
    print("Archiving model version {}".format(model.version))
    client.transition_model_version_stage(
      name=config['model']['name'],
      version=int(model.version),
      stage="Archived"
    )

# COMMAND ----------

client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name=config['model']['name'],
    version=version,
    stage="Production"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Merchant similarity
# MAGIC Here comes the moment we were all waiting for. Could a model carefully designed to learn words based on sentences be ported to the world of card transactions and learn merchants based on their customer base? Most importantly, could customers be segmented by the type of shops they visit? If that assumption stands true, we would have built a data asset that could be used for a variety of use cases in retail banking, from pricing, targeting, cross-sell, upsell opportunities as well as advanced fraud prevention strategies. 

# COMMAND ----------

import mlflow
pipeline = mlflow.spark.load_model("models:/{}/production".format(config['model']['name']))
word2Vec_model = pipeline.stages[0]

# COMMAND ----------

# MAGIC %md
# MAGIC With no ground truth around merchant categories, the obvious way to quickly validate our approach is to eyeball its results and apply domain expertise. Personally a fan of brands like "Paul Smith", our model can find Paul Smiths' closest competitors to be "Hugo Boss", "Ralph Lauren" or "Tommy Hilfinger". This first test has proven to be successful. Most importantly, our model did not simply detect brands within the same category (fashion industry), but would appear to detect brands of similar price tags, exhibiting a pattern that, if validated, would exceed our expectations. Not only could we classify lines of businesses, but customer segmentation could be also driven by the quality of goods they purchase. The same was also observed by Capital One in their excellent [white paper](https://arxiv.org/pdf/1907.07225.pdf). 
# MAGIC 
# MAGIC <img src='https://i.pinimg.com/originals/90/cc/6b/90cc6b771a52fe5bba3521a44a0f8da6.jpg' width=100>
# MAGIC <img src='https://2.bp.blogspot.com/-W76pRH63G9s/UzOzhXY0n_I/AAAAAAAAcac/juZFMKMWoyY/s1600/james-franco-gucci-made-to-measure.jpg' width=150>
# MAGIC <img src='https://m.media-amazon.com/images/I/91XHl6VuShL._SL1500_.jpg' width=100>
# MAGIC <img src='http://4.bp.blogspot.com/-TTkLAX7MFJ8/UZNdoUbO0zI/AAAAAAABGq0/kFu-qFAXUo8/s1600/TH_FR_single_spread2.jpg' width=150>

# COMMAND ----------

display(
  word2Vec_model
    .findSynonyms('Paul Smith', 5)
    .withColumnRenamed('word', 'merchant_name')
)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's see if our model could pick up different "lines of businesses", or charities in this case. Customers regularly donating to charities could exhibit different spending behaviors than others. In this case, the closest synonyms of "British Red Cross" would be "medecin sans frontieres", "save the children" or "RSPCA".
# MAGIC 
# MAGIC <img src='https://www.parkacademyboston.net/wp-content/uploads/2019/07/NSPCC.png' width=100>
# MAGIC <img src='https://pbs.twimg.com/profile_images/1071486462049304576/LGlSNB2K_400x400.jpg' width=90>
# MAGIC <img src='https://yt3.ggpht.com/ytc/AKedOLQeZeb03LJtcZcDvQ2fjecLvjizxjTFxvngIBhanA=s900-c-k-c0x00ffffff-no-rj' width=100>
# MAGIC <img src='https://blogs.msf.org/sites/default/files/styles/author/public/default_images/msf-default-logo_0.jpg?itok=3N7S6iOg' width=100>

# COMMAND ----------

display(
  word2Vec_model
    .findSynonyms('British Red Cross', 5)
    .withColumnRenamed('word', 'merchant_name')
)

# COMMAND ----------

# MAGIC %md
# MAGIC Gambling activities are often a sensitive subject in retail banking. Although statistically significant for credit risk decisioning, leveraging such characteristics may be unethical (when not simply illegal in certain regulated countries), just like gender or ethnicity. Whilst we could see our model picking up on gambling activities that are similar, and (sadly) not so dissimilar to pawn shops, small loans, or liquor shops, it could actually be used for good, when put in good hands. One could decide to ignore those activities leaving everyone with a fair and ethical access to consumer credit whilst others could leverage these patterns to offer more personalized advice such as finance coaching or debt consolidation that would actually help their end users.
# MAGIC 
# MAGIC <img src='https://static.perform.news/sites/2/2021/02/24044050/Ladbrokes-Logo.png' width=100>
# MAGIC <img src='https://pbs.twimg.com/profile_images/1000461740772134913/T9-zMXmF_400x400.jpg' width=100>
# MAGIC <img src='https://uploads-ssl.webflow.com/5e6f7cd3ee7f51d539a4da0b/5f85674b0221b7c2c225a362_pr_source.png' width=100>
# MAGIC <img src='https://pbs.twimg.com/profile_images/1422505290398830646/GfAMN8i6_400x400.jpg' width=100>

# COMMAND ----------

display(
  word2Vec_model
    .findSynonyms('Betfred', 5)
    .withColumnRenamed('word', 'merchant_name')
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Merchant classification
# MAGIC The few examples above were surprisingly troubling to say the least, but we certainly do not know all brands and their similarities to declare success. There might be groups of merchants more or less similar than others that we may want to identify further. The easiest way to find those significant groups of merchants / brands is to visualize our embedded vector space. For that purpose, we would need to apply techniques like [Principal Component Analysis](https://royalsocietypublishing.org/doi/10.1098/rsta.2015.0202) (PCA) to reduce these 255 large vectors into 3 dimensions.

# COMMAND ----------

import numpy as np
from pyspark.sql.functions import udf
import pyspark.sql.functions as F
from pyspark.ml.functions import vector_to_array

merchant_vectors = (
  word2Vec_model
    .getVectors()
    .withColumnRenamed('word', 'merchant_name')
    .withColumn('merchant_vector', vector_to_array('vector'))
    .select('merchant_name', 'merchant_vector')
)

# COMMAND ----------

df = merchant_vectors.toPandas()
X = np.array(list(df.merchant_vector))

# COMMAND ----------

import pandas as pd
from sklearn.decomposition import PCA

# not a model per se, we do not wish to log it onto mlflow
# PCA is used here simply for visualization purpose
mlflow.sklearn.autolog(disable=True)

pca = PCA(n_components=3).fit_transform(X)
pca_df = pd.DataFrame(data = pca, columns = ['c1', 'c2', 'c3'])
pca_df['merchant_name'] = df.merchant_name

# COMMAND ----------

import plotly.express as px

xaxis = dict(
             backgroundcolor="rgb(200, 200, 230)",
             gridcolor="white",
             showbackground=True,
             zerolinecolor="white"
)

yaxis = dict(
             backgroundcolor="rgb(200, 230, 230)",
             gridcolor="white",
             showbackground=True,
             zerolinecolor="white"
)

zaxis = dict(
             backgroundcolor="rgb(230, 230, 230)",
             gridcolor="white",
             showbackground=True,
             zerolinecolor="white"
)

fig = px.scatter_3d(
  pca_df,
  x='c1',
  y='c2',
  z='c3',
  hover_name='merchant_name', 
  width=800, 
  height=600, 
  opacity=0.6
)

fig.update_traces(marker_size = 3)
fig.update_layout(scene = dict(xaxis = xaxis, yaxis = yaxis, zaxis = zaxis))
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Using a simple 3D plot, we could identify 5 distinct groups of merchants. These merchants may be different lines of business, may even be dissimilar at first glance (and definitely different as per their MCC codes), but have something in common: they all attract a similar customer base. For instance, customers mostly shopping in gambling and low cost brands may differ from those shopping for luxury items and organic food. We can confirm this hypothesis through a clustering model (KMeans).

# COMMAND ----------

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

# broadcast features so that workers can access efficiently
X_broadcast = sc.broadcast(X)
 
# function to train model and return metrics
def evaluate_model(n):
  model = KMeans( n_clusters=n, init='k-means++', n_init=1, max_iter=10000)
  clusters = model.fit(X_broadcast.value).labels_
  return n, float(model.inertia_), float(silhouette_score(X_broadcast.value, clusters))

# define number of iterations for each value of k being considered
iterations = (
  spark
    .range(100) # iterations per value of k
    .crossJoin( spark.range(2,21).withColumnRenamed('id','n')) # cluster counts
    .repartition(config['model']['exec'])
    .select('n')
    .rdd
    )
 
# train and evaluate model for each iteration
results_pd = (
  spark
    .createDataFrame(
      iterations.map(lambda n: evaluate_model(n[0])), # iterate over each value of n
      schema=['n', 'inertia', 'silhouette']
      ).toPandas()
    )
 
# remove broadcast set from workers
X_broadcast.unpersist()

# COMMAND ----------

# MAGIC %md
# MAGIC Plotting KMeans inertia relative to the target number of clusters, we can see that the total sum of squared distances between cluster members and cluster centers decreases as we increase the number of clusters. Our goal is not to drive inertia to zero (which would be achieved if we made each member the center of its own) but instead to identify the point in the curve where the incremental drop in inertia is diminished. In our plot, we might identify this point as occurring somewhere between 5 and 10 clusters, just like what we've spotted earlier through our 3D plot.

# COMMAND ----------

import matplotlib.pyplot as plt
results_pd = results_pd.sort_values(by="n")
plt.bar(results_pd.n, results_pd.silhouette)

# COMMAND ----------

k = 7
mlflow.sklearn.autolog(disable=False)

with mlflow.start_run(run_name='merchcat') as run:
  run_id = run.info.run_id
  merchcat_model = KMeans(n_clusters=k, init='k-means++', n_init=1, max_iter=10000).fit(X)
  y_pred = pd.Series(merchcat_model.predict(X))

# COMMAND ----------

pca_df['cluster'] = y_pred.apply(lambda x: 'merchcat-{}'.format(x))

fig = px.scatter_3d(
  pca_df,
  x='c1',
  y='c2',
  z='c3',
  hover_name='merchant_name', 
  color='cluster',
  width=800, 
  height=600, 
  opacity=0.6
)

fig.update_traces(marker_size = 3)
fig.update_layout(scene = dict(xaxis = xaxis, yaxis = yaxis, zaxis = zaxis))
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Finally, we have grouped our merchants into 7 categories that attract the same customer personas, moving from a traditional approach made of industry standards (MCC) to a more accurate representation based on actual customer spending behavior, hence a better candidate for modern customer segmentation use cases.

# COMMAND ----------

from scipy import spatial
cluster_centers = pd.DataFrame([[i, c] for i, c in enumerate(merchcat_model.cluster_centers_)], columns=['merchant_cluster', 'cluster_centroid'])
distance_to_center = lambda x: spatial.distance.euclidean(x.merchant_vector, x.cluster_centroid)

# Attach cluster to each merchant
df['merchant_cluster'] = y_pred

# Compute distance from every point to its centroid
merged_vectors = df.merge(right_on='merchant_cluster', left_on='merchant_cluster', right=cluster_centers)
merged_vectors['distance_to_centroid'] = merged_vectors.apply(distance_to_center, axis=1)

# COMMAND ----------

_ = (
  spark.createDataFrame(merged_vectors)
    .select('merchant_name', 'merchant_vector', 'merchant_cluster', 'distance_to_centroid')
    .write
    .format('delta')
    .mode('overwrite')
    .save('{}/embeddings'.format(home_dir))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Customer segmentation
# MAGIC Although we had a bit of fun applying a model out-of-its original box, we did not really address our key challenge of modern customer segmentation. To get back to our NLP analogy, we were able to learn the meaning of words, but not documents. In our case, we have learned the meaning of merchants and brands, but not customer behaviors. One of the odd features of the `word2vec` model is that sufficiently large vectors could still be aggregated whilst maintaining high predictive value. To put it another way, the significance of a document could be learnt by averaging the vector of each of its constituents (see [whitepaper](https://arxiv.org/pdf/1405.4053v2.pdf) from the creators of `word2vec`, Tomas Mikolov, et al.). Similarly, we will learn customer spending preferences by aggregating vectors of each of their preferred brands. Two customers having similar taste for luxury brands, high end cars and liquor would theoretically be close from one another, hence belonging to the same segment.

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

customer_merchants = (
  transactions_raw
    .join(spark.read.format('delta').load('{}/embeddings'.format(home_dir)), ['merchant_name'])
    .groupBy('customer_id')
    .agg(F.collect_list('merchant_name').alias('walks'))
)

customer_embeddings = (
  word2Vec_model
    .transform(customer_merchants)
    .drop('walks')
)

# COMMAND ----------

# MAGIC %md
# MAGIC It is worth mentioning that such an aggregated view would generate a sort of transactional fingerprint that is unique to each of our end consumers. Although two fingerprints may share similar traits (same preferences), these signatures can be used to track customer unique behaviors, **over time**. When signature drastically differs from previous observations, this could be a sign of fraudulent activities (sudden interest for gambling companies). When signature drifts over time, this could be indicative of life events (having a new born kid). This approach would be key to drive hyper-personalization in retail banking, tracking customer preferences over time and become the go-to banks across various life events, positive or negative.

# COMMAND ----------

customer_embeddings_df = customer_embeddings.withColumn('embedding', vector_to_array('embedding')).toPandas()
X = np.array(list(customer_embeddings_df.embedding))

# COMMAND ----------

# not a model per se, we do not wish to log it onto mlflow
# PCA is used here simply for visualization purpose
mlflow.sklearn.autolog(disable=True)

pca = PCA(n_components=3).fit_transform(X)
pca_df = pd.DataFrame(data = pca, columns = ['c1', 'c2', 'c3'])
pca_df['customer_id'] = customer_embeddings_df['customer_id']
pca_df = pca_df.sample(n=10000)

# COMMAND ----------

# MAGIC %md
# MAGIC Similar to our merchant visualizations, we can represent each customers' fingerprints into a 3D plane using principal component analysis. Although we observe the vast majority of users closely packed together, we may identify specific behaviors stretched across our 3 dimensions.

# COMMAND ----------

fig = px.scatter_3d(
  pca_df,
  x='c1',
  y='c2',
  z='c3',
  width=800, 
  height=600, 
  opacity=0.6
)

fig.update_traces(marker_size = 2)
fig.update_layout(scene = dict(xaxis = xaxis, yaxis = yaxis, zaxis = zaxis))
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC At this point, given the indisputable predictive potential offered by this data asset, we recommend this excellent [solution accelerator](https://databricks.com/solutions/accelerators/customer-segmentation) from our retail counterpart, Bryan Smith, technical director for retail and CPG at Databricks, who walks us through different segmentation techniques used by best in class retail organizations. We invite readers to go through this retail solution to find different approaches and techniques to clustering. But for now, let's define 5 shopping personas through a simple KMeans model.

# COMMAND ----------

from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

with mlflow.start_run(run_name='segmentation') as run:

  run_id = run.info.run_id

  # Trains a k-means model
  kmeans = KMeans().setK(5).setSeed(42).setFeaturesCol('embedding')
  kmeans_model = kmeans.fit(customer_embeddings)

  # Make predictions
  predictions = kmeans_model.transform(customer_embeddings)

  # Evaluate clustering by computing Silhouette score
  evaluator = ClusteringEvaluator().setFeaturesCol('embedding')
  evaluator.evaluate(predictions)

# COMMAND ----------

cohort_df = (
  transactions_raw
    .join(spark.read.format('delta').load('{}/embeddings'.format(home_dir)), ['merchant_name'])
    .withColumn('merchant_cluster', F.concat(F.lit('merch_cat_'), F.col('merchant_cluster')))
    .groupBy('customer_id', 'merchant_cluster')
    .count()
    .join(predictions, ['customer_id'])
    .withColumnRenamed('prediction', 'cohort')
    .orderBy('cohort')
    .groupBy('cohort', 'merchant_cluster')
    .agg(F.avg('count').alias('avg_count'))
    .select(
      F.col('cohort'),
      F.col('avg_count').alias('average_visits'),
      F.col('merchant_cluster').alias('merchant_category')
    )
    .orderBy('cohort')
)

# COMMAND ----------

display(cohort_df)

# COMMAND ----------

# MAGIC %md
# MAGIC As represented above, our 5 clusters exhibit different spending behaviors. Whilst cluster #0 seems to be biased towards gambling activities, our cluster #4 is more centered around online businesses and subscription based services, probably indicative of a younger generation of customers. We invite our readers to complement this view with what they already know about their customers (original segments, products and services, average income, demographics, etc.) to better understand each of those behavioral driven segments. 

# COMMAND ----------

from pyspark.sql.window import Window

prev_x = (
  Window
    .partitionBy(F.col('customer_id'))
    .orderBy(F.col('date'))
    .rowsBetween(-50, 0)
)

window_transactions = (
  transactions_raw
    .join(word2Vec_model.getVectors().select(F.col('word').alias('merchant_name')), ['merchant_name'])
    .withColumn('walks', F.collect_list('merchant_name').over(prev_x))
)

# COMMAND ----------

# MAGIC %md
# MAGIC As briefly introduced earlier, we could leverage that same data asset to detect changes over time. By applying a sliding window, we could compare previous fingerprints for a given customer and track changes, over time. Using a simple cosine similarity, one may detect sudden drifts possibly indicative of fraudulent activities. New features could be generated daily and injected to online fraud prevention strategies as introduced in a different [solution accelerator](https://databricks.com/solutions/accelerators/fraud-detection), combining rules + AI. We represent below the digital banking signatures of 5 random users over a course of 1 year. 

# COMMAND ----------

from scipy import spatial

@F.udf('float')
def cosine(x1, x2):
  return 1 - float(spatial.distance.cosine(x1, x2))

customer_timeseries = (
  word2Vec_model
    .transform(window_transactions)
    .select('date', 'customer_id', 'merchant_name', 'embedding')
    .withColumn('previous', F.lag(F.col('embedding')).over(Window.partitionBy('customer_id').orderBy('date')))
    .filter(F.col('previous').isNotNull())
    .withColumn('previous', vector_to_array('previous'))
    .withColumn('embedding', vector_to_array('embedding'))
    .withColumn('similarity', cosine(F.col('previous'), F.col('embedding')))
    .groupBy('customer_id', 'date')
    .agg(F.avg('similarity').alias('similarity'))
    .orderBy('date')
    .cache()
)

# COMMAND ----------

import plotly.express as px

top_5 = customer_timeseries.groupBy('customer_id').count().orderBy(F.desc('count')).limit(5).toPandas().customer_id.tolist()
timeseries = customer_timeseries.where(F.col('customer_id').isin(top_5)).toPandas()
df = timeseries.pivot(index='date', columns='customer_id', values='similarity').fillna(method='ffill')

fig = px.line(df, x=df.index, y=df.columns, width=1100, height=500)
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Closing Thoughts
# MAGIC In this solution accelerator, we have borrowed a few concepts from the world of Natural Language Processing that we successfully ported out to card transactions for customer segmentation in retail banking. We also demonstrated the relevance of the Lakehouse for Financial Services to address this problem of scale where graph analytics, matrix computation, natural language processing, clustering techniques must be combined into one platform. Compared to traditional segmentation methods in the world of SQL, the future of segmentation can only be addressed through data+AI, at scale. 
# MAGIC 
# MAGIC Although we appreciate we've only scratched the surface of what was possible using off-the-shelf models and data at our disposal, we proved that hyper-personalization could be driven by customer spending patterns better than demographics, opening up a exciting range of new opportunities from cross sell / upsell / pricing / targeting activities as well as Fraud detection strategies. Most importantly, this technique allowed us to learn from new-to-bank individuals or individuals without a known credit history by leveraging information from others. With 55 million underbanked in the US in 2018 according to the federal reserve ([source](https://www.federalreserve.gov/publications/2019-economic-well-being-of-us-households-in-2018-banking-and-credit.htm)), such an approach could pave the way towards a more customer centric, inclusive and ethical future of retail banking. 
