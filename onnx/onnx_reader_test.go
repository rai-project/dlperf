package onnx

import (
	"path/filepath"
	"testing"

	"github.com/GeertJohan/go-sourcepath"
	"github.com/stretchr/testify/assert"
)

func TestOnnxReader(t *testing.T) {
	cwd := sourcepath.MustAbsoluteDir()
	onnxModelFile := filepath.Join(cwd, "..", "assets", "onnx_models", "mnist.onnx")

	// model, err := onnx.ReadModelShapeInfer(onnxModelFile)
	// assert.NoError(t, err)
	// assert.NotEmpty(t, model)

	model, err := NewOnnx(onnxModelFile)
	assert.NoError(t, err)
	assert.NotEmpty(t, model)

	graph := model.GetGraph()
	nodes := graph.Node
	for _, n := range nodes {
		if n.GetOpType() == "Conv" {
			model.GetNodeInputDimensions(n)
		}
	}

	// info := model.FlopsInformation()
	// pp.Println(info)
	// pp.Println(net.LayerInformations())

	// net.Layer = nil
	// net.Layers = nil
	// pp.Println(net)
}
