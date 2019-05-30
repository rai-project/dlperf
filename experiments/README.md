## Experiments of inference using onnx models

### Install Requirements

Use `pipenv` to launch a shell

```
pipenv shell
```

Or manually install the packages using pip

```
pip install onnx gluoncv mxnet onnxmltools onnxruntime torchvision click pycodestyle torch tensorflow future onnx-tf tvm
```

### Run

```
python main.py --debug --backend=mxnet
```
