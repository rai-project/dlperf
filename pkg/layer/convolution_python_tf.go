package layer

import "github.com/go-python/gpython/ast"

func (c Conv) FwdPythonTensorflowAST() (ast.Ast, error) {
	prog := &ast.Module{}
	prog.Body = []ast.Stmt{
		&ast.Import{
			Names: []*ast.Alias{
				&ast.Alias{
					Name:   "numpty",
					AsName: "np",
				},
			},
		},
		&ast.Import{
			Names: []*ast.Alias{
				&ast.Alias{
					Name: "os",
				},
			},
		},
		&ast.Import{
			Names: []*ast.Alias{
				&ast.Alias{
					Name: "sys",
				},
			},
		},
		&ast.Import{
			Names: []*ast.Alias{
				&ast.Alias{
					Name:   "tensorflow",
					AsName: "tf",
				},
			},
		},
		&ast.ImportFrom{
			Module: "common.params",
			Names: []*ast.Alias{
				&ast.Alias{
					Name: "*",
				},
			},
		},
		&ast.ImportFrom{
			Module: "common.utils",
			Names: []*ast.Alias{
				&ast.Alias{
					Name: "*",
				},
			},
		},
	}

	return prog, nil
}

func (c Conv) FwdPythonTensorflow() (string, error) {
	a, err := c.FwdPythonTensorflowAST()
	if err != nil {
		return "", err
	}
	return ast.Dump(a), nil
}
