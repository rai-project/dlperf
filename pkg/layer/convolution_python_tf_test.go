package layer

import (
	"testing"

	"github.com/k0kubun/pp"
	"github.com/stretchr/testify/assert"
)

func TestConvolutionPythonTF(t *testing.T) {
	conv := Conv{}

	prog, err := conv.FwdPythonTensorflow()
	assert.NoError(t, err)
	assert.NotEmpty(t, prog)
	pp.Println(prog)
}
