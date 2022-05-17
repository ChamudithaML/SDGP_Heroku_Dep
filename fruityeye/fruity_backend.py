# importing required python libraries
import os
from flask import Flask, request, render_template, send_from_directory, Response
from fruityeye import app

# import tensorflow as tf
# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     try:
#         # Currently, memory growth needs to be the same across GPUs
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         # Memory growth must be set before GPUs have been initialized
#         print(e)


# route to home page
@app.route('/home')
@app.route('/')
def home_page():
    return render_template('home.html')


# route to login page
@app.route('/login')
def login_page():
    return render_template('login.html')

# route to image gallery page
@app.route('/imageGallery')
def imageGallery_page():
    return render_template('imageGallery.html')

# ------------------------------------------------------------------