import argparse
import json
import os.path
import sys
import click
import glob


import info
import utils
import input_image
from models import models


def get_backend(backend):
    utils.debug("Loading {} backend".format(backend))
    if backend == "tensorflow":
        from backend_tf import BackendTensorflow

        backend = BackendTensorflow()
    if backend == "caffe2":
        from backend_caffe2 import BackendCaffe2

        backend = BackendCaffe2()
    elif backend == "onnxruntime":
        from backend_onnxruntime import BackendOnnxruntime

        backend = BackendOnnxruntime()
    elif backend == "null":
        from backend_null import BackendNull

        backend = BackendNull()
    elif backend == "pytorch":
        from backend_pytorch import BackendPytorch

        backend = BackendPytorch()
    elif backend == "pytorch-native":
        from backend_pytorch_native import BackendPytorchNative

        backend = BackendPytorchNative()
    elif backend == "tflite":
        from backend_tflite import BackendTflite

        backend = BackendTflite()
    else:
        raise ValueError("unknown backend: " + backend)
    return backend


# @click.option(
#     "-d",
#     "--debug",
#     type=click.BOOL,
#     is_flag=True,
#     help="print debug messages to stderr.",
#     default=False,
# )
# @click.option(
#     "-q",
#     "--quiet",
#     type=click.BOOL,
#     is_flag=True,
#     help="don't print messages",
#     default=False,
# )


@click.command()
@click.option("--backend", type=click.STRING, default="onnxruntime")
@click.option(
    "--debug/--no-debug", help="print debug messages to stderr.", default=False
)
@click.option("--quiet/--no-quiet", help="don't print messages", default=False)
@click.pass_context
@click.version_option()
def main(ctx, backend, debug, quiet):
    model = models[1]

    utils.DEBUG = debug
    utils.QUIET = quiet
    backend = get_backend(backend)

    img = input_image.get(model)

    backend.load(model)
    t = backend.forward(img)

    print("elapsed time = {}".format(t))


if __name__ == "__main__":
    main()
