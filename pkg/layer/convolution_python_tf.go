package layer

import (
	"bytes"
	"errors"
	"fmt"

	"github.com/go-python/gpython/ast"
	"github.com/go-python/gpython/parser"
	"github.com/go-python/gpython/py"
	"github.com/rai-project/dlperf/pkg"
)

func createPythonTFTensor(name string, ty string, shape dlperf.Shape) *ast.Assign {
	tensor := &ast.Assign{
		Targets: []ast.Expr{
			&ast.Name{
				Id: ast.Identifier(name),
			},
		},
		Value: &ast.Call{
			Func: &ast.Name{Id: ast.Identifier("tf.placeholder")},
			Args: []ast.Expr{
				&ast.Name{
					Id: ast.Identifier(ty),
				},
			},
			Keywords: []*ast.Keyword{
				&ast.Keyword{
					Arg: ast.Identifier("shape"),
					Value: &ast.List{
						Elts: []ast.Expr{
							&ast.Num{
								N: py.Int(shape[0]),
							},
							&ast.Num{
								N: py.Int(shape[1]),
							},
							&ast.Num{
								N: py.Int(shape[2]),
							},
							&ast.Num{
								N: py.Int(shape[3]),
							},
						},
					},
				},
			},
		},
	}

	return tensor
}

func (c Conv) FwdPythonTensorflowAST() (ast.Ast, error) {

	args := c.FwdBenchmarkArgs().(convBenchmarkArgs)

	header := _escFSMustString(dlperf.IsDebug, "/tensorflow/base_prefix.py")
	prog0, err := parser.Parse(bytes.NewBufferString(header), "test", "exec")
	if err != nil {
		return nil, err
	}

	prog, ok := prog0.(*ast.Module)
	if !ok {
		return nil, errors.New("expecting module")
	}

	x := createPythonTFTensor(
		"x",
		"tf.float32",
		dlperf.Shape{
			args.Input0,
			args.Input1,
			args.Input2,
			args.Input3,
		},
	)

	conv := &ast.Assign{
		Targets: []ast.Expr{
			&ast.Name{
				Id: ast.Identifier(fmt.Sprintf("conv_%d", args.UniqueBenchmarkID)),
			},
		},
		Value: &ast.Call{
			Func: &ast.Name{Id: ast.Identifier("tf.layers.conv2d")},
			Args: []ast.Expr{
				x.Targets[0],
				&ast.Num{
					N: py.Int(args.FilterCount),
				},
				&ast.List{
					Elts: []ast.Expr{
						&ast.Num{
							N: py.Int(args.FilterHeight),
						},
						&ast.Num{
							N: py.Int(args.FilterWidth),
						},
					},
				},
			},
			Keywords: []*ast.Keyword{
				&ast.Keyword{
					Arg: ast.Identifier("dilation_rate"),
					Value: &ast.List{
						Elts: []ast.Expr{
							&ast.Num{
								N: py.Int(args.DilationHeight),
							},
							&ast.Num{
								N: py.Int(args.DilationWidth),
							},
						},
					},
				},
				&ast.Keyword{
					Arg: ast.Identifier("strides"),
					Value: &ast.List{
						Elts: []ast.Expr{
							&ast.Num{
								N: py.Int(args.StrideHeight),
							},
							&ast.Num{
								N: py.Int(args.StrideWidth),
							},
						},
					},
				},
				&ast.Keyword{
					Arg: ast.Identifier("padding"),
					Value: &ast.Str{
						S: py.String(c.AutoPad),
					},
				},
			},
		},
	}

	prog.Body = append(prog.Body, x)
	prog.Body = append(prog.Body, conv)

	return prog, nil
}

func (c Conv) FwdPythonTensorflow() (string, error) {
	a, err := c.FwdPythonTensorflowAST()
	if err != nil {
		return "", err
	}
	// return ast.Dump(a), nil
	return ToPythonString(a)
}
