//go:generate go get -v github.com/mailru/easyjson/...
//go:generate go get github.com/valyala/quicktemplate/qtc
//go:generate go get github.com/gotpl/gtfmt
//go:generate go get -v github.com/mjibson/esc
//go:generate go get github.com/wlbr/templify
//go:generate esc -o generated_data.go -pkg layer -prefix codegen -private codegen
//go:generate easyjson -disallow_unknown_fields -build_tags !debug -snake_case -pkg .
package layer
