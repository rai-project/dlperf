import mxnet as mx
from mxnet.contrib import onnx as onnx_mxnet
import time
from collections import namedtuple

import backend


class BackendMXNet(backend.Backend):
    def __init__(self):
        super(BackendMXNet, self).__init__()
        self.session = None
        self.input_data = None
        self.input_name = None
        self.ctx = mx.gpu() if len(mx.test_utils.list_gpus()) else mx.cpu()

    def name(self):
        return "mxnet"

    def version(self):
        return mxnet.__version__

    def load(self, model):
        self.sym, self.arg, self.aux = onnx_mxnet.import_model(model.path)
        self.data_names = [
            graph_input
            for graph_input in self.sym.list_inputs()
            if graph_input not in self.arg and graph_input not in self.aux
        ]
        self.model = mx.mod.Module(
            symbol=self.sym,
            data_names=self.data_names,
            context=self.ctx,
            label_names=None,
        )

    def forward_once(self, img):
        start = time.time()
        result = self.model.forward(img)
        end = time.time()  # stop timer
        return end - start

    def forward(self, img, warmup=True):
        Batch = namedtuple("Batch", ["data"])
        img = mx.nd.array(img, ctx=self.ctx)
        self.model.bind(
            for_training=False,
            data_shapes=[(self.data_names[0], img.shape)],
            label_shapes=None,
        )
        self.model.set_params(
            arg_params=self.arg,
            aux_params=self.aux,
            allow_missing=True,
            allow_extra=True,
        )
        img = Batch([img])
        if warmup:
            self.forward_once(img)
        return self.forward_once(img)

