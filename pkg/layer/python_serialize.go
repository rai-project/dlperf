package layer

import (
	"bytes"
	"fmt"

	"github.com/go-python/gpython/ast"
)

func ToPythonString(a ast.Ast) (string, error) {
	var walk func(a ast.Ast)

	buf := bytes.NewBufferString("")

	walkStmts := func(stmts []ast.Stmt) {
		for _, stmt := range stmts {
			walk(stmt)
			buf.WriteString("\n")
		}
	}
	_ = walkStmts
	walkExprs := func(exprs []ast.Expr) {
		for _, expr := range exprs {
			walk(expr)
		}
	}
	_ = walkExprs

	writeIdentifier := func(id ast.Identifier) {
		buf.WriteString(fmt.Sprintf("%s", string(id)))
	}

	walk = func(a ast.Ast) {
		if a == nil {
			return
		}
		switch node := a.(type) {
		case *ast.Module:
			walkStmts(node.Body)
		case *ast.Alias:
			writeIdentifier(node.Name)
			if node.AsName != "" {
				buf.WriteString(" as ")
				writeIdentifier(node.AsName)
			}
		case *ast.Import:
			buf.WriteString("import ")
			for _, n := range node.Names {
				walk(n)
			}
		case *ast.ImportFrom:
			buf.WriteString("from ")
			writeIdentifier(node.Module)
			buf.WriteString(" import ")
			for ii, n := range node.Names {
				walk(n)
				if ii < len(node.Names)-1 {
					buf.WriteString(", ")
				}
			}
		case *ast.Attribute:
			walk(node.Value)
			if node.Attr != "" {
				buf.WriteString(".")
				writeIdentifier(node.Attr)
			}
		case *ast.Num:
			buf.WriteString(fmt.Sprintf("%v", node.N))
		case *ast.List:
			buf.WriteString("[")
			for ii, n := range node.Elts {
				walk(n)
				if ii < len(node.Elts)-1 {
					buf.WriteString(", ")
				}
			}
			buf.WriteString("]")
		case *ast.Call:
			walk(node.Func)
			buf.WriteString("(")
			for ii, n := range node.Args {
				walk(n)
				if ii < len(node.Args)-1 {
					buf.WriteString(", ")
				}
			}
			if len(node.Keywords) != 0 {
				buf.WriteString(", ")
			}
			for ii, n := range node.Keywords {
				writeIdentifier(n.Arg)
				buf.WriteString("=")
				walk(n.Value)
				if ii < len(node.Keywords)-1 {
					buf.WriteString(", ")
				}
			}
			buf.WriteString(")")
		case *ast.ExprStmt:
			walk(node.Value)
		case *ast.Name:
			writeIdentifier(node.Id)
		case *ast.Slice:
			walk(node.Lower)
			buf.WriteString(":")
			walk(node.Upper)
			buf.WriteString(":")
			walk(node.Step)
		case *ast.Index:
			walk(node.Value)
		case *ast.Subscript:
			walk(node.Value)
			buf.WriteString("[")
			walk(node.Slice)
			buf.WriteString("]")
		case *ast.Str:
			buf.WriteString(fmt.Sprintf("'%s'", string(node.S)))
		case *ast.Assign:
			for ii, expr := range node.Targets {
				walk(expr)
				if ii < len(node.Targets)-1 {
					buf.WriteString(", ")
				}
			}
			buf.WriteString(" = ")
			walk(node.Value)
		default:
			fmt.Printf("%T\n", a)
		}

	}

	walk(a)

	return buf.String(), nil
}
