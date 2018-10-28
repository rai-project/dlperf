import numpy as np
import os
import sys
import tensorflow as tf
from common.params import *
from common.utils import *
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print("OS: ", sys.platform)
print("Python: ", sys.version)
print("Numpy: ", np.__version__)
print("Tensorflow: ", tf.__version__)
print("GPU: ", get_gpu_name())
print(get_cuda_version())
print("CuDNN Version ", get_cudnn_version())
