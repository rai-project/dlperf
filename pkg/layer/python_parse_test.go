package layer

import (
	"bytes"
	"testing"

	"github.com/go-python/gpython/parser"
	"github.com/k0kubun/pp"
	"github.com/stretchr/testify/assert"
)

func TestPythonParse(t *testing.T) {
	const testProg = `
import numpy as np
import os
import sys
import tensorflow as tf
from common.params import *
from common.utils import *
slim = tf.contrib.slim`

	// parser.SetDebug(4)
	tree, err := parser.Parse(bytes.NewBufferString(testProg), "test", "exec")
	assert.NoError(t, err)
	assert.NotEmpty(t, tree)
	pp.Println(tree)
}

// &ast.Module{
//   ModBase: ast.ModBase{
//     Pos: ast.Pos{
//       Lineno:    0,
//       ColOffset: 0,
//     },
//   },
//   Body: []ast.Stmt{
//     &ast.Import{
//       StmtBase: ast.StmtBase{
//         Pos: ast.Pos{
//           Lineno:    2,
//           ColOffset: 0,
//         },
//       },
//       Names: []*ast.Alias{
//         &ast.Alias{
//           Pos: ast.Pos{
//             Lineno:    2,
//             ColOffset: 7,
//           },
//           Name:   "numpy",
//           AsName: "np",
//         },
//       },
//     },
//     &ast.Import{
//       StmtBase: ast.StmtBase{
//         Pos: ast.Pos{
//           Lineno:    3,
//           ColOffset: 0,
//         },
//       },
//       Names: []*ast.Alias{
//         &ast.Alias{
//           Pos: ast.Pos{
//             Lineno:    3,
//             ColOffset: 7,
//           },
//           Name:   "os",
//           AsName: "",
//         },
//       },
//     },
//     &ast.Import{
//       StmtBase: ast.StmtBase{
//         Pos: ast.Pos{
//           Lineno:    4,
//           ColOffset: 0,
//         },
//       },
//       Names: []*ast.Alias{
//         &ast.Alias{
//           Pos: ast.Pos{
//             Lineno:    4,
//             ColOffset: 7,
//           },
//           Name:   "sys",
//           AsName: "",
//         },
//       },
//     },
//     &ast.Import{
//       StmtBase: ast.StmtBase{
//         Pos: ast.Pos{
//           Lineno:    5,
//           ColOffset: 0,
//         },
//       },
//       Names: []*ast.Alias{
//         &ast.Alias{
//           Pos: ast.Pos{
//             Lineno:    5,
//             ColOffset: 7,
//           },
//           Name:   "tensorflow",
//           AsName: "tf",
//         },
//       },
//     },
//     &ast.ImportFrom{
//       StmtBase: ast.StmtBase{
//         Pos: ast.Pos{
//           Lineno:    6,
//           ColOffset: 0,
//         },
//       },
//       Module: "common.params",
//       Names:  []*ast.Alias{
//         &ast.Alias{
//           Pos: ast.Pos{
//             Lineno:    6,
//             ColOffset: 26,
//           },
//           Name:   "*",
//           AsName: "",
//         },
//       },
//       Level: 0,
//     },
//     &ast.ImportFrom{
//       StmtBase: ast.StmtBase{
//         Pos: ast.Pos{
//           Lineno:    7,
//           ColOffset: 0,
//         },
//       },
//       Module: "common.utils",
//       Names:  []*ast.Alias{
//         &ast.Alias{
//           Pos: ast.Pos{
//             Lineno:    7,
//             ColOffset: 25,
//           },
//           Name:   "*",
//           AsName: "",
//         },
//       },
//       Level: 0,
//     },
//     &ast.Assign{
//       StmtBase: ast.StmtBase{
//         Pos: ast.Pos{
//           Lineno:    8,
//           ColOffset: 0,
//         },
//       },
//       Targets: []ast.Expr{
//         &ast.Name{
//           ExprBase: ast.ExprBase{
//             Pos: ast.Pos{
//               Lineno:    8,
//               ColOffset: 0,
//             },
//           },
//           Id:  "slim",
//           Ctx: 2,
//         },
//       },
//       Value: &ast.Attribute{
//         ExprBase: ast.ExprBase{
//           Pos: ast.Pos{
//             Lineno:    8,
//             ColOffset: 17,
//           },
//         },
//         Value: &ast.Attribute{
//           ExprBase: ast.ExprBase{
//             Pos: ast.Pos{
//               Lineno:    8,
//               ColOffset: 9,
//             },
//           },
//           Value: &ast.Name{
//             ExprBase: ast.ExprBase{
//               Pos: ast.Pos{
//                 Lineno:    8,
//                 ColOffset: 7,
//               },
//             },
//             Id:  "tf",
//             Ctx: 1,
//           },
//           Attr: "contrib",
//           Ctx:  1,
//         },
//         Attr: "slim",
//         Ctx:  1,
//       },
//     },
//   },
// }
