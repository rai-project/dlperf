package cmd

import (
	"bytes"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"strings"

	"github.com/k0kubun/pp"

	"github.com/Unknwon/com"
	"github.com/olekukonko/tablewriter"
	"github.com/rai-project/dlperf/pkg/writer"
)

type Writer struct {
	output         io.Writer
	outputFileName string
	tbl            *tablewriter.Table
	csv            *csv.Writer
	json           []string
	tex           []string
	opts           []writer.Option
}

type Rower interface {
	Header(opts ...writer.Option) []string
	Row(opts ...writer.Option) []string
}

func NewWriter(rower Rower, iopts ...writer.Option) *Writer {

	iopts = append([]writer.Option{writer.Format(outputFormat)}, iopts...)

	var output io.Writer = os.Stdout
	if outputFileName != "" {
		output = &bytes.Buffer{}
	}
	wr := &Writer{
		output:         output,
		outputFileName: outputFileName,
		opts:           iopts,
	}
	opts := writer.NewOptions(iopts...)
	switch opts.Format {
	case "table":
		wr.tbl = tablewriter.NewWriter(output)
		// make it markdown format
		wr.tbl.SetBorders(tablewriter.Border{Left: true, Top: false, Right: true, Bottom: false})
		wr.tbl.SetCenterSeparator("|")
	case "latex", "tex":
		wr.tex = []string{}
	case "csv":
		wr.csv = csv.NewWriter(output)
	case "json":
		wr.json = []string{}
	}
	if rower != nil && (!noHeader || appendOutput) {
		wr.Header(rower)
	}
	return wr
}

func (w *Writer) Header(rower Rower) error {
	opts := writer.NewOptions(w.opts...)
	switch opts.Format {
	case "table":
		w.tbl.SetHeader(rower.Header(w.opts...))
	case "latex", "tex":
		var r  string
		header :=  rower.Header(w.opts...)
		for ii, entry := range header {
			r += `\thead{\textbf{` + texEscape(entry) + `}}`
			if ii == len(header) {
				r  += ` \\`
			} else {
				r  += ` & `
			}
		}
		w.tex = append(w.tex, r)
	case "csv":
		w.csv.Write(rower.Header(w.opts...))
	}
	return nil
}


func texEscape(str string)string {
	text := []byte(str)
	out := &bytes.Buffer{}
	for i := 0; i < len(text); i++ {
		// directly copy normal characters
		org := i

		for i < len(text) && !texNeedsBackslash(text[i]) {
			i++
		}
		if i > org {
			out.Write(text[org:i])
		}

		// escape a character
		if i >= len(text) {
			break
		}
		out.WriteByte('\\')
		out.WriteByte(text[i])
	}
	return string(out.Bytes())	
}


func texNeedsBackslash(c byte) bool {
	for _, r := range []byte("_{}%$&\\~") {
		if c == r {
			return true
		}
	}
	return false
}

func (w *Writer) Row(rower Rower) error {
	opts := writer.NewOptions(w.opts...)
	switch opts.Format {
	case "table":
		w.tbl.Append(rower.Row(w.opts...))
	case "latex", "tex":
		var r  string
		row := rower.Row(w.opts...)
		for ii, entry := range row {
			r += texEscape(entry)
			if ii == len(row) {
				r  += ` \\`
			} else {
				r  += ` & `
			}
		}
		w.tex = append(w.tex, r)
	case "csv":
		w.csv.Write(rower.Row(w.opts...))
	case "json", "js":
		b, err := json.MarshalIndent(rower, "", "  ")
		if err != nil {
			log.WithError(err).Error("failed to marshal json data...")
			return err
		}

		b = bytes.Replace(b, []byte("\\u003c"), []byte("<"), -1)
		b = bytes.Replace(b, []byte("\\u003e"), []byte(">"), -1)
		b = bytes.Replace(b, []byte("\\u0026"), []byte("&"), -1)

		if false {
			pp.Println(rower)
		}
		w.json = append(w.json, string(b))
	}
	return nil
}

func (w *Writer) Flush() {
	opts := writer.NewOptions(w.opts...)
	switch opts.Format {
	case "table":
		w.tbl.Render()
	case "csv":
		w.csv.Flush()
	case "tex", "latex":
		r := ""
		prevData := ""
		if com.IsFile(w.outputFileName) && appendOutput {
			buf, err := ioutil.ReadFile(w.outputFileName)
			if err == nil {
				prevData = string(buf)
			}
		}
		prevData = prevData
		r += strings.Join(w.tex, "\n")
		w.output.Write([]byte(r))
	case "json":
		prevData := ""
		if com.IsFile(w.outputFileName) && appendOutput {
			buf, err := ioutil.ReadFile(w.outputFileName)
			if err == nil {
				prevData = string(buf)
			}
		}
		prevData = strings.TrimSpace(prevData)
		js := "["
		if prevData != "" && prevData != "[]" {
			js += strings.TrimSuffix(strings.TrimPrefix(prevData, "["), "]")
			js += ",\n"
		}
		js += strings.Join(w.json, ",\n")
		js += "]"
		w.output.Write([]byte(js))
	}
}

func (w *Writer) Close() {
	w.Flush()
	if w.outputFileName != "" {
		com.WriteFile(w.outputFileName, w.output.(*bytes.Buffer).Bytes())
		return
	}
	opts := writer.NewOptions(w.opts...)
	format := opts.Format
	if format == "json" {
		fmt.Println(w.json)
	}
}
