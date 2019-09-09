package cmd

import (
	"fmt"
	"path/filepath"
	"strings"

	"github.com/pkg/errors"
	"github.com/rai-project/config"
	dlperf "github.com/rai-project/dlperf/pkg"
	"github.com/rai-project/dlperf/pkg/writer"
	"github.com/spf13/cast"
	"github.com/spf13/cobra"
)

var (
	allYears              = []int{2010, 2012, 2013, 2014, 2015, 2016, 2017, 2018}
	year              int = -1    // all years
	lessThanEqualYear     = false // toggle less than equal on year
)

type layerYearInfo struct {
	Year   int `json:"year"`
	Total  int `json:"total"`
	Unique int `json:"unique"`
}

func (layerYearInfo) Header(opts ...writer.Option) []string {
	return []string{"Year", "Total", "Unique"}
}

func (l layerYearInfo) Row(iopts ...writer.Option) []string {
	return []string{cast.ToString(l.Year), cast.ToString(l.Total), cast.ToString(l.Unique)}
}

func uniqueLayersPerYear(year int, commulative bool) (*layerYearInfo, error) {
	layersOf := func(modelPath string) dlperf.Layers {
		models, err := readModels(modelPath)
		if err != nil {
			return nil
		}

		var layers dlperf.Layers
		for _, model := range models {
			model.Information() // no assignment on purpose
			nds, err := model.Network().TopologicallyOrderedNodes()
			if err != nil {
				return nil
			}
			for _, nd := range nds {
				opType := strings.ToLower(nd.Layer().OperatorType())
				if opType == "constant_input" || opType == "constantinput" || opType == "constant" {
					continue
				}
				layers = append(layers, nd.Layer())
			}
		}
		return layers
	}

	models := []modelURLInfo{}
	for _, m := range modelURLs {
		if m.Year == year {
			models = append(models, m)
			continue
		}
		if commulative && m.Year < year {
			models = append(models, m)
			continue
		}
	}
	modelPaths, err := getModelsIn(modelPath)
	if err != nil {
		return nil, errors.Wrapf(err, "unable to glob %s", modelPath)
	}
	var layers dlperf.Layers
out:
	for _, m := range models {
		for _, path := range modelPaths {
			modelName := getModelName(path)
			if modelName == m.Name {
				lyrs := layersOf(path)
				if lyrs == nil {
					panic("expecting a list of layers...")
				}
				layers = append(layers, lyrs...)
				continue out
			}
		}
		fmt.Printf("unable to find path for %v\n", m.Name)
	}

	totalLayers := len(layers)

	layers = layers.FwdUnion("float", "")

	uniqueLayers := len(layers)

	return &layerYearInfo{
		Year:   year,
		Total:  totalLayers,
		Unique: uniqueLayers,
	}, nil
}

var layerYearInfoCmd = &cobra.Command{
	Use:     "year_info",
	Aliases: []string{"year"},
	Short:   "Generates information on number of layers and number of unique layers for a year",
	PreRunE: func(cmd *cobra.Command, args []string) error {
		if modelPath == "automatic" || modelPath == "" {
			modelPath = filepath.Join(config.App.TempDir, "dlperf")
		}
		return nil
	},
	RunE: func(cmd *cobra.Command, args []string) error {
		years := []int{}
		if year == -1 {
			years = allYears
		} else if lessThanEqualYear == true {
			for _, yr := range allYears {
				if yr <= year {
					years = append(years, yr)
				}
			}
		} else {
			years = append(years, year)
		}

		writer := NewWriter(
			layerYearInfo{},
		)
		defer writer.Close()

		commulative := lessThanEqualYear
		for _, yr := range years {
			info, err := uniqueLayersPerYear(yr, commulative)
			if err != nil {
				panic(err)
			}
			writer.Row(info)
		}
		return nil
	},
}

func init() {
	layerYearInfoCmd.PersistentFlags().BoolVar(&lessThanEqualYear, "commulative", false, "consider all years less than equal specified year")
	layerYearInfoCmd.PersistentFlags().IntVar(&year, "year", -1, "get info for particular year (-1 for all years)")
	rootCmd.AddCommand(layerYearInfoCmd)
}
