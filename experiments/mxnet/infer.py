import numpy as np
import mxnet as mx
from mxnet.contrib import onnx as onnx_mxnet
from mxnet import gluon, nd
import tarfile, os
import json
import logging
logging.basicConfig(level=logging.INFO)

image_folder = "images"
utils_file = "utils.py" # contain utils function to plot nice visualization
image_net_labels_file = "image_net_labels.json"
images = ['apron.jpg', 'hammerheadshark.jpg', 'dog.jpg', 'wrench.jpg', 'dolphin.jpg', 'lotus.jpg']
base_url = "https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/doc/tutorials/onnx/{}?raw=true"

for image in images:
    mx.test_utils.download(base_url.format("{}/{}".format(image_folder, image)), fname=image,dirname=image_folder)
mx.test_utils.download(base_url.format(utils_file), fname=utils_file)
mx.test_utils.download(base_url.format(image_net_labels_file), fname=image_net_labels_file)

from utils import *

current_model = "bvlc_googlenet"
model_folder = "model"
archive = "{}.tar.gz".format(current_model)
archive_file = os.path.join(model_folder, archive)
url = "{}{}".format(base_url, archive)
mx.test_utils.download(url, dirname = model_folder)

if not os.path.isdir(os.path.join(model_folder, current_model)):
    print('Extracting model...')
    tar = tarfile.open(archive_file, "r:gz")
    tar.extractall(model_folder)
    tar.close()
    print('Extracted')
