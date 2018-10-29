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
	templateBasePrefix           string
	templateBaseSuffix           string
	templateBaseStandalonePrefix string
	templateBaseStandaloneSuffix string
)

func init() {
	templateBasePrefix = _escFSMustString(false, "/scope/base_prefix.tmpl")
	templateBaseSuffix = _escFSMustString(false, "/scope/base_suffix.tmpl")
	templateBaseStandalonePrefix = _escFSMustString(false, "/scope/base_standalone_prefix.tmpl")
	templateBaseStandaloneSuffix = _escFSMustString(false, "/scope/base_standalone_suffix.tmpl")
}

// recovery will silently swallow all unexpected panics.
func recovery() {
	recover()
}

func templateExec(lyr dlperf.Layer, templString string) string {
	tmpl, err := mkTemplate(lyr).Parse(templString)
	if err != nil {
		panic(err)
	}
	args := lyr.FwdBenchmarkArgs()
	if args == nil {
		return ""
	}
	buf := bytes.NewBufferString("")
	err = tmpl.Execute(buf, args)
	if err != nil {
		panic(err)
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

// e.g.
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
//     {"unique_benchmark_id", [[.Id]]} /* UniqueBenchmarkID */
func mkTemplateCounters(st interface{}) string {
	defer recovery()
	id := ""
	res := []string{}
	for _, field := range structs.New(st).Fields() {
		if field.IsExported() && structs.IsStruct(field.Value()) {
			args := mkTemplateCounters(field.Value())
			if len(args) > 0 {
				res = append(res, args)
			}
			continue
		}
		idTag := field.Tag("id")
		if idTag == "true" {
			tag := field.Tag("json")
			id = fmt.Sprintf(`      {"%s", %v} /* %s */, `, tag, field.Value(), field.Name())
		}
		tag := field.Tag("args")
		if tag == "" || tag == "-" {
			continue
		}
		res = append(res, fmt.Sprintf(`      {"%s", %v} /* %s */, `, tag, field.Value(), field.Name()))
	}
	if id != "" {
		res = append(res, id)
	}
	return strings.Join(res, "\n")
}

// e.g.
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
// [[.Id]] /* UniqueBenchmarkID */ \
func mkTemplateArguments(st interface{}) string {
	defer recovery()
	id := ""
	res := []string{}
	for _, field := range structs.New(st).Fields() {
		if field.IsExported() && structs.IsStruct(field.Value()) {
			args := mkTemplateArguments(field.Value())
			if len(args) > 0 {
				res = append(res, args)
			}
			continue
		}

		idTag := field.Tag("id")
		if idTag == "true" {
			id = fmt.Sprintf(`      %v /* %s */, \`, field.Value(), field.Name())
		}
		tag := field.Tag("args")
		if tag == "" || tag == "-" {
			continue
		}
		e := fmt.Sprintf(`      %v /* %s */, \`, field.Value(), field.Name())
		res = append(res, e)
	}
	if id != "" {
		res = append(res, id)
	}
	return strings.Join(res, "\n")
}
