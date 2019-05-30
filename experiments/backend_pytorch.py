import torch
import onnx
import time
import caffe2.python.onnx.backend

import backend


class BackendPytorch(backend.Backend):
    def __init__(self):
        super(BackendPytorch, self).__init__()
        self.session = None
        self.input_data = None
        self.input_name = None

    def name(self):
        return "pytroch"

    def version(self):
        return torch.__version__

    def load(self, model):
        self.model = onnx.load(model.path)
        self.inputs = []
        initializers = set()
        for i in self.model.graph.initializer:
            initializers.add(i.name)
        for i in self.model.graph.input:
            if i.name not in initializers:
                self.inputs.append(i.name)
        self.outputs = []
        for i in self.model.graph.output:
            self.outputs.append(i.name)
        device = "CUDA:0" if torch.cuda.is_available() else "CPU"
        self.session = caffe2.python.onnx.backend.prepare(self.model, device)

    def forward_once(self, img):
        start = time.time()
        result = self.session.run(self.input_data)
        end = time.time()  # stop timer
        return end - start

    def forward(self, img, warmup=True):
        if warmup:
            self.forward_once(img)
        return self.forward_once(img)
