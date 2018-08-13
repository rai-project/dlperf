package onnx

import "fmt"

func (o Onnx) String() string {
	// https://github.com/okdshin/instant/blob/master/tool/onnx_viewer.cpp
	return ""
}

func (o Onnx) Show() {
	fmt.Println(o.String())
}
