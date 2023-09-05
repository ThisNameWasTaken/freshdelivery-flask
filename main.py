import json

from typing import Dict, Text

import pandas as pd
import numpy as np
import tensorflow as tf

import tensorflow_recommenders as tfrs

from flask import Flask, jsonify
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app)

@app.route("/", methods=['GET'])
@cross_origin()
def home():
   return "hello world"

orders = pd.read_csv('./orders.csv', dtype={'user_id': "string", 'product_id': "string"})
products = orders.groupby(["product_id"]).size().reset_index(name="purchase_count")


orders = pd.read_csv('./orders.csv', dtype={'user_id': "string", 'product_id': "string"})
products = orders.groupby(["product_id"]).size().reset_index(name="purchase_count")
orders = orders.groupby(["user_id", "product_id"]).size().reset_index(name="order_count")
orders = orders.sample(frac=1)
# orders
orders = tf.data.Dataset.from_tensor_slices(dict(orders))
products = tf.data.Dataset.from_tensor_slices(dict(products))

ratings = orders.map(lambda x: {
    "product_id": x["product_id"],
    "user_id": x["user_id"]
})

movies = products.map(lambda x: x["product_id"])

user_ids_vocabulary = tf.keras.layers.StringLookup(mask_token=None)
user_ids_vocabulary.adapt(ratings.map(lambda x: x["user_id"]))

movie_titles_vocabulary = tf.keras.layers.StringLookup(mask_token=None)
movie_titles_vocabulary.adapt(movies)

class MovieLensModel(tfrs.Model):
  # We derive from a custom base class to help reduce boilerplate. Under the hood,
  # these are still plain Keras Models.

  def __init__(
      self,
      user_model: tf.keras.Model,
      movie_model: tf.keras.Model,
      task: tfrs.tasks.Retrieval):
    super().__init__()

    # Set up user and movie representations.
    self.user_model = user_model
    self.movie_model = movie_model

    # Set up a retrieval task.
    self.task = task

  def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
    # Define how the loss is computed.

    user_embeddings = self.user_model(features["user_id"])
    movie_embeddings = self.movie_model(features["product_id"])

    return self.task(user_embeddings, movie_embeddings)
  
# Define user and movie models.
user_model = tf.keras.Sequential([
    user_ids_vocabulary,
    tf.keras.layers.Embedding(user_ids_vocabulary.vocab_size(), 64)
])

movie_model = tf.keras.Sequential([
    movie_titles_vocabulary,
    tf.keras.layers.Embedding(movie_titles_vocabulary.vocab_size(), 64)
])

# Define your objectives.
# task = tfrs.tasks.Retrieval(metrics=tfrs.metrics.FactorizedTopK(
#     movies.batch(128).map(movie_model)
#   )
# )
# Remove metrics when saving
task = tfrs.tasks.Retrieval()

# Create a retrieval model.
model = MovieLensModel(user_model, movie_model, task)
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.5))

# Train for 3 epochs.
model.fit(ratings.batch(4096), epochs=7)

index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
index.index_from_dataset(
    movies.batch(100).map(lambda title: (title, model.movie_model(title))))

# TFRS setup


@app.route("/recommended-products/<user_id>", methods=['GET'])
@cross_origin()
def recommended_products(user_id):
    # _, product_ids = index(tf.constant([user_id]))
    _, product_ids = index(np.array([user_id]))
    arr = np.vectorize(lambda x: x.decode())(product_ids.numpy().ravel()).tolist()
    print(arr)
    return jsonify(arr)

app.run(host="0.0.0.0")