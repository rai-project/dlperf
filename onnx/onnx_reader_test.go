package onnx

import (
	"path/filepath"
	"testing"

	"github.com/GeertJohan/go-sourcepath"
	"github.com/stretchr/testify/assert"
)

func TestOnnxReader(t *testing.T) {
	cwd := sourcepath.MustAbsoluteDir()
	mnistProtoTxtPath := filepath.Join(cwd, "..", "assets", "onnx_models", "mnist.onnx")
	model, err := NewOnnx(mnistProtoTxtPath)
	assert.NoError(t, err)
	assert.NotEmpty(t, model)

	// info := model.FlopsInformation()
	// pp.Println(info)
	// pp.Println(net.LayerInformations())

	// net.Layer = nil
	// net.Layers = nil
	// pp.Println(net)
}
