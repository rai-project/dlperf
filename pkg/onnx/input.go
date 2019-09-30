package onnx

import (
	"github.com/pkg/errors"
)

func getInputName(modelPath string) (string, error) {
	name := getModelName(modelPath)

	switch name {
	case "DenseNet-121":
		return "data_0", nil
	case "Inception-v1":
		return "data_0", nil
	}

	return "", errors.New("pick first layer")
}
