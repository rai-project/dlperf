package layer

import (
	"os"
	"testing"

	"github.com/alecthomas/chroma/quick"
	dlperf "github.com/rai-project/dlperf/pkg"
	"github.com/stretchr/testify/assert"
)

func TestConvolutionPythonTF(t *testing.T) {
	conv := Conv{
		Base: &Base{
			InputShapes_: []dlperf.Shape{
				dlperf.Shape{
					1, 3, 224, 224,
				},
				dlperf.Shape{
					11, 11, 3, 96,
				},
			},
			OutputShapes_: []dlperf.Shape{
				dlperf.Shape{
					1, 55, 55, 96,
				},
			},
		},
		AutoPad: "valid",
		KernelShape: dlperf.Shape{
			11, 11,
		},
		Pads: dlperf.Shape{
			0, 0, 0, 0,
		},
		Strides: dlperf.Shape{
			4, 4,
		},
		Dilations: dlperf.Shape{
			0, 0,
		},
	}

	prog, err := conv.FwdPythonTensorflow()
	assert.NoError(t, err)
	assert.NotEmpty(t, prog)

	err = quick.Highlight(os.Stdout, prog, "python", "terminal256", "monokai")
	assert.NoError(t, err)
}
