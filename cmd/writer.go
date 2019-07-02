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

	"github.com/360EntSecGroup-Skylar/excelize"

	"github.com/k0kubun/pp"

	"github.com/Unknwon/com"
	"github.com/olekukonko/tablewriter"
)

type excel struct {
	*excelize.File
	sheet int
}

type Writer struct {
	format         string
	output         io.Writer
	outputFileName string
	tbl            *tablewriter.Table
	csv            *csv.Writer
	excel          *excel
	json           []string
	humanFlops     bool
}

type Rower interface {
	Header() []string
	Row(humanFlops bool) []string
}

func NewWriter(rower Rower, humanFlops bool) *Writer {
	var output io.Writer = os.Stdout
	if outputFileName != "" {
		output = &bytes.Buffer{}
	}
	wr := &Writer{
		format:         outputFormat,
		output:         output,
		outputFileName: outputFileName,
		humanFlops:     humanFlops,
	}
	switch wr.format {
	case "table":
		wr.tbl = tablewriter.NewWriter(output)
	case "csv":
		wr.csv = csv.NewWriter(output)
	case "xslt", "excel":
		ex := excelize.NewFile()
		wr.excel = &excel{
			File:  ex,
			sheet: ex.NewSheet("Sheet1"),
		}
	case "json":
		wr.json = []string{}
	}
	if rower != nil && (!noHeader || appendOutput) {
		wr.Header(rower)
	}
	return wr
}

func (w *Writer) Header(rower Rower) error {
	switch w.format {
	case "table":
		w.tbl.SetHeader(rower.Header())
	case "csv":
		w.csv.Write(rower.Header())
	}
	return nil
}

func (w *Writer) Row(rower Rower) error {
	switch w.format {
	case "table":
		w.tbl.Append(rower.Row(w.humanFlops))
	case "csv":
		w.csv.Write(rower.Row(w.humanFlops))
	case "json", "js":
		buf, err := json.MarshalIndent(rower, "", "  ")
		if err != nil {
			log.WithError(err).Error("failed to marshal json data...")
			return err
		}
		if false {
			pp.Println(rower)
		}
		w.json = append(w.json, string(buf))
	}
	return nil
}

func (w *Writer) Flush() {
	switch w.format {
	case "table":
		w.tbl.Render()
	case "csv":
		w.csv.Flush()
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
	if w.format == "json" {
		fmt.Println(w.json)
	}
}
