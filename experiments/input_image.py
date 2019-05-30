import numpy as np
from PIL import Image


def preprocess_imagenet(img_data):
    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(img_data.shape).astype("float32")
    for i in range(img_data.shape[0]):
        # for each pixel in each channel, divide the value by 255 to get value between [0, 1] and then normalize
        norm_img_data[i, :, :] = (img_data[i, :, :] / 255 - mean_vec[i]) / stddev_vec[i]
    return norm_img_data


# def get(model):
#     np.random.seed(0)
#     return np.asarray(np.random.uniform(model.shape), dtype=np.float32)


def get(model):
    input = np.asarray(Image.open("images/dog.jpg"))
    input = np.asarray([input[:, :, 0], input[:, :, 1], input[:, :, 2]])

    input_wrapped = []
    input_wrapped.append(input)
    input_wrapped = np.asarray(input_wrapped)
    return preprocess_imagenet(input_wrapped)

