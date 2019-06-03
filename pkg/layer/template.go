package layer

import (
	"bytes"
	"fmt"
	"strings"
	"text/template"

	"github.com/fatih/structs"
	"github.com/leekchan/gtf"
	dlperf "github.com/rai-project/dlperf/pkg"
)

var (
	templateBasePrefix string
	templateBaseSuffix string
)

func init() {
	templateBasePrefix = _escFSMustString(false, "/scope/base_prefix.tmpl")
	templateBaseSuffix = _escFSMustString(false, "/scope/base_suffix.tmpl")
}

// recovery will silently swallow all unexpected panics.
func recovery() {
	recover()
}

func templateExecFWD(lyr dlperf.Layer, templString string, opts ...dlperf.FwdBenchmarkArgsOptionFunc) string {
	tmpl, err := mkTemplate(lyr).Parse(templString)
	if err != nil {
		log.Fatal(err)
	}
	args := lyr.FwdBenchmarkArgs(opts...)
	if args == nil {
		return ""
	}
	buf := bytes.NewBufferString("")
	err = tmpl.Execute(buf, args)
	if err != nil {
		log.Fatal(err)
	}
	return buf.String()
}

func templateExecBWD(lyr dlperf.Layer, templString string, opts ...dlperf.BwdBenchmarkArgsOptionFunc) string {
	tmpl, err := mkTemplate(lyr).Parse(templString)
	if err != nil {
		log.Fatal(err)
	}
	args := lyr.BwdBenchmarkArgs(opts...)
	if args == nil {
		return ""
	}
	buf := bytes.NewBufferString("")
	err = tmpl.Execute(buf, args)
	if err != nil {
		log.Fatal(err)
	}
	return buf.String()
}

func copy(originalMap map[string]interface{}) map[string]interface{} {
	newMap := make(map[string]interface{})
	for key, value := range originalMap {
		newMap[key] = value
	}
	return newMap
}

func mkTemplate(lyr dlperf.Layer) *template.Template {
	funcs := copy(gtf.GtfTextFuncMap)
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
		if field.IsExported() && structs.IsStruct(field.Value()) {
			args := mkTemplateCounters(field.Value())
			if len(args) > 0 {
				res = append(res, args)
			}
			continue
		}
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
	ii := 0
	for _, field := range structs.New(st).Fields() {
		if field.IsExported() && structs.IsStruct(field.Value()) {
			args := mkTemplateArguments(field.Value())
			if len(args) > 0 {
				res = append(res, args)
			}
			continue
		}
		tag := field.Tag("args")
		if tag == "" || tag == "-" {
			continue
		}
		res = append(res, fmt.Sprintf(`      %v /* %s , idx = %d*/, \`, field.Value(), field.Name(), ii))
		ii++
	}
	return strings.Join(res, "\n")
}
