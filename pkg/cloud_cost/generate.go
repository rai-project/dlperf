//go:generate go get -v github.com/mailru/easyjson/...
//go:generate go get github.com/valyala/quicktemplate/qtc
//go:generate go get github.com/gotpl/gtfmt
//go:generate go get -v github.com/mjibson/esc
//go:generate go get github.com/wlbr/templify
//go:generate esc -o generated_data.go -pkg cloud_cost -prefix _fixtures -private _fixtures
//go:generate easyjson -disallow_unknown_fields -snake_case -pkg .

package cloud_cost
