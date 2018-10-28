//go:generate go get -v github.com/mailru/easyjson/...
//go:generate easyjson -disallow_unknown_fields -snake_case -pkg .
//go:generate go get github.com/valyala/quicktemplate/qtc
//go:generate go get github.com/gotpl/gtfmt
//go:generate go get -v github.com/mjibson/esc
//go:generate go get github.com/wlbr/templify
//xxx go:generate esc -o generated_templates.go -pkg layer -prefix codegen -private codegen/scope

package layer
