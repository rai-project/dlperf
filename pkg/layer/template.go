package layer

import (
	"fmt"
	"strings"
	"text/template"

	"github.com/fatih/structs"
	"github.com/leekchan/gtf"
	dlperf "github.com/rai-project/dlperf/pkg"
)

// recovery will silently swallow all unexpected panics.
func recovery() {
	recover()
}

func mkTemplate(lyr dlperf.Layer) *template.Template {
	funcs := gtf.GtfTextFuncMap
	funcs["make_counters"] = mkTemplateCounters
	funcs["make_arguments"] = mkTemplateArguments
	return template.New(lyr.OperatorType()).
		Funcs(funcs).
		Delims("[[", "]]")
}

// {"input[0]", [[.Input0]]}, /* Input0 */ \
//     {"input[1]", [[.Input1]]}, /* Input1 */ \
//     {"input[2]", [[.Input2]]}, /* Input2 */ \
//     {"input[3]", [[.Input3]]}, /* Input3 */ \
//     {"filter_count", [[.FilterCount]]}, /* FilterCount */ \
//     {"filter_height", [[.FilterHeight]]}, /* FilterHeight */ \
//     {"filter_width", [[.FilterWidth]]}, /* FilterWidth */ \
//     {"pad_height", [[.PadHeight]]}, /* PadHeight */ \
//     {"pad_width", [[.PadWidth]]}, /* PadWidth */ \
//     {"stride_height", [[.StrideHeight]]}, /* StrideHeight */ \
//     {"stride_width", [[.StrideWidth]]}, /* StrideWidth */ \
//     {"dilation_height", [[.DilationHeight]]}, /* DilationHeight */ \
//     {"dilation_width", [[.DilationWidth]]} /* DilationWidth */
func mkTemplateCounters(st interface{}) string {
	defer recovery()
	res := []string{}
	for _, field := range structs.New(st).Fields() {
		tag := field.Tag("args")
		if tag == "" || tag == "-" {
			continue
		}
		res = append(res, fmt.Sprintf(`      {"%s", %v} /* %s */, `, tag, field.Value(), field.Name()))
	}
	return strings.Join(res, "\n")
}

// [[.Input0]], /* Input0 */ \
// [[.Input1]], /* Input1 */ \
// [[.Input2]], /* Input2 */ \
// [[.Input3]], /* Input3 */ \
// [[.FilterCount]], /* FilterCount */ \
// [[.FilterHeight]], /* FilterHeight */ \
// [[.FilterWidth]], /* FilterWidth */ \
// [[.PadHeight]], /* PadHeight */ \
// [[.PadWidth]], /* PadWidth */ \
// [[.StrideHeight]], /* StrideHeight */ \
// [[.StrideWidth]], /* StrideWidth */ \
// [[.DilationHeight]], /* DilationHeight */ \
// [[.DilationWidth]] /* DilationWidth */ \
func mkTemplateArguments(st interface{}) string {
	defer recovery()
	res := []string{}
	for _, field := range structs.New(st).Fields() {
		tag := field.Tag("args")
		if tag == "" || tag == "-" {
			continue
		}
		res = append(res, fmt.Sprintf(`      %v /* %s */, \`, field.Value(), field.Name()))
	}
	return strings.Join(res, "\n")
}
