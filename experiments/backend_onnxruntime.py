import onnxruntime
import time
import backend

import utils


class BackendOnnxruntime(backend.Backend):
    def __init__(self):
        super(BackendOnnxruntime, self).__init__()
        self.session = None

    def name(self):
        return "onnxruntime"

    def version(self):
        return onnxruntime.__version__

    def load(self, model):
        self.model = model
        self.session = onnxruntime.InferenceSession(model.path, None)
        self.inputs = [meta.name for meta in self.session.get_inputs()]
        self.outputs = [meta.name for meta in self.session.get_outputs()]
        utils.debug("inputs of onnxruntime is {}".format(self.inputs))
        utils.debug("outputs of onnxruntime is {}".format(self.outputs))

    def forward_once(self, img):
        start = time.time()
        result = self.session.run(self.outputs, {self.inputs[0]: img})
        end = time.time()  # stop timer
        return end - start

    def forward(self, img, warmup=True):
        if warmup:
            self.forward_once(img)
        return self.forward_once(img)
