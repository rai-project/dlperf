## Experiments of inference using onnx models

### Install Requirements

Use [`pipenv`](https://github.com/pypa/pipenv) to launch a shell

```
pipenv shell
```

if you get an error you may need to install

```
pipenv install --python=`pyenv which python`
```

Or manually install the packages using pip

```
pip install onnx gluoncv mxnet onnxmltools onnxruntime torchvision click pycodestyle torch tensorflow future onnx-tf tvm
```


### Run

```
python main.py --debug --backend=mxnet
```
