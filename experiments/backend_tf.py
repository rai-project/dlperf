import onnx
import warnings

warnings.filterwarnings("ignore")

import os
import utils

# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from onnx_tf.backend import prepare
from onnx_tf.common import supports_device
import tensorflow as tf
import time

import backend


class BackendTensorflow(backend.Backend):
    def __init__(self):
        super(BackendTensorflow, self).__init__()
        self.session = None
        self.input_data = None
        self.input_name = None
        utils.debug("getting device = {}".format(supports_device("CUDA")))
        self.device = "/device:GPU:0" if supports_device("CUDA") else "/cpu:0"

    def name(self):
        return "tensorflow"

    def version(self):
        return tf.__version__

    def load(self, model):
        utils.debug("loading onnx model {} from disk".format(model.path))
        self.onnx_model = onnx.load(model.path)
        with tf.device(self.device):
            self.model = prepare(onnx_model)
        self.session = tf.Session(graph=self.model)

    def forward_once(self, img):
        with torch.no_grad():
            start = time.time()
            result = self.model(img)
            end = time.time()  # stop timer
            return end - start

    def forward(self, img, warmup=True):
        img = torch.tensor(img).float().to(self.device)
        if warmup:
            self.forward_once(img)
        return self.forward_once(img)
