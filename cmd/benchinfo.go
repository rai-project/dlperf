package cmd

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime/debug"
	"strings"
	"time"

	sourcepath "github.com/GeertJohan/go-sourcepath"
	"github.com/Unknwon/com"
	"github.com/alecthomas/repr"
	"github.com/k0kubun/pp"
	"github.com/pkg/errors"
	"github.com/spf13/cobra"

	"github.com/rai-project/config"
	dlperf "github.com/rai-project/dlperf/pkg"
	"github.com/rai-project/dlperf/pkg/benchmark"
	perflayer "github.com/rai-project/dlperf/pkg/layer"
	"github.com/rai-project/dlperf/pkg/onnx"
	"gonum.org/v1/gonum/graph/encoding/dot"
	"gonum.org/v1/gonum/graph/path"
)

var (
	benchmarkResultsFolder      string
	benchInfoTraining           bool
	benchInfoShort              bool
	benchInfoDataType           string
	enableReadFlopsFromDatabase bool
	traversalStrategy           string
	showBenchInfo               bool
	defaultTraversalStrategy    = "parallel"
)

func benchinfo(cmd *cobra.Command, args []string) error {
	defer func() {
		if r := recover(); r != nil {
			fmt.Println(string(debug.Stack()))
			pp.Println("[PANIC] while computing bench information " + modelPath + " [error = " + repr.String(r) + "]")
		}
	}()

	if com.IsDir(modelPath) {
		baseOutputFileName := outputFileName
		if !com.IsDir(baseOutputFileName) {
			os.MkdirAll(baseOutputFileName, os.ModePerm)
		}
		modelPaths, err := getModelsIn(modelPath)
		if err != nil {
			return errors.Wrapf(err, "unable to glob %s", modelPath)
		}
		progress := newProgress("> Computing bench info of models", len(modelPaths))
		defer progress.Finish()
		for _, path := range modelPaths {
			progress.Increment()
			modelPath = path
			modelName := getModelName(modelPath)
			outputFileName = filepath.Join(baseOutputFileName, modelName+"."+outputFormat)
			if true {
				pp.Println("processing " + modelName + " from " + modelPath + " to " + outputFileName)
			}
			if err := benchinfo(cmd, args); err != nil {
				pp.Println("failed processing "+modelName+" from "+modelPath+" to "+outputFileName, errors.WithStack(err))
			}
		}
		return nil
	}

	modelPath = expandModelPath(modelPath)

	dlperf.SetBatchSize(batchSize)

	benchSuite, err := benchmark.New(benchmarkResultsFolder)
	if err != nil {
		return err
	}

	model, err := onnx.New(modelPath, batchSize)
	if err != nil {
		return err
	}

	net := model.Network()

	nds, err := net.TopologicallyOrderedNodes()
	if err != nil {
		return err
	}

	benchmarkInfo := []benchmarkGraphNode{}

	totalTime := time.Duration(0)
	totalFlops := dlperf.FlopsInformation{}

	debugPrint := func(val ...interface{}) {
		if config.App.IsDebug {
			fmt.Println(val...)
		}
	}

	for _, nd := range nds {
		lyr := nd.Layer()
		switch strings.ToLower(lyr.OperatorType()) {
		case "constantinput", "lrn", "reshape", "concat", "unsqueeze", "flatten", "globalpooling", "identity", "transpose", "scale":
			benchmarkInfo = append(benchmarkInfo,
				benchmarkGraphNode{
					GraphNode: nd,
				},
			)

			debugPrint(lyr.OperatorType() + " layer is skipped for now")
			continue
		}
		// if lyr.OperatorType() != "Conv" && lyr.OperatorType() != "Relu" {
		// 	pp.Println(lyr.OperatorType())
		// 	continue
		// }

		filterBenchmarks := func(benchInfoBackward bool, datatype string, algorithm string, iopts ...interface{}) *benchmark.Benchmark {
			if benchInfoBackward {
				opts := make([]dlperf.BwdBenchmarkArgsOptionFunc, len(iopts))
				for ii, opt := range iopts {
					opts[ii] = opt.(dlperf.BwdBenchmarkArgsOptionFunc)
				}
				filter := lyr.BwdBenchmarkFilter(datatype, algorithm, opts...)
				return &filter
			}

			opts := make([]dlperf.FwdBenchmarkArgsOptionFunc, len(iopts))
			for ii, opt := range iopts {
				opts[ii] = opt.(dlperf.FwdBenchmarkArgsOptionFunc)
			}
			filter := lyr.FwdBenchmarkFilter(datatype, algorithm, opts...)
			return &filter
		}

		getBenchmarkTime := func(filter *benchmark.Benchmark) (benchmark.Benchmarks, error) {
			if filter == nil {
				pp.Println(lyr.Name())
				pp.Println(filter)
				return nil, errors.New("empty filter")
			}
			bs, err := benchSuite.Filter(*filter)
			if err != nil {
				// pp.ColoringEnabled = false
				// log.WithError(err).WithField("filter", pp.Sprint(filter)).Error("failed to find benchmark within benchmark suite")
				// pp.ColoringEnabled = true
				// continue

				pp.Println(lyr.Name())
				pp.Println(filter)
				return nil, errors.New("invalid filter benchmarks")
			}
			if len(bs) == 0 {
				// pp.ColoringEnabled = false
				// log.WithField("filter", pp.Sprint(filter)).Error("unable to find benchmark within benchmark suite")
				// pp.ColoringEnabled = true
				// continue
				// pp.Println(lyr.OperatorType())
				// pp.Println(lyr.Name())
				pp.Println(filter)
				return nil, errors.New("no benchmarks")
			}
			return bs, nil
		}

		makeLayerInfos := func(bs benchmark.Benchmarks, ty string) *bench {
			bs.Sort()

			flops := lyr.Information().Flops()
			if enableReadFlopsFromDatabase {
				if bs[0].Flops != nil && *bs[0].Flops != -1 {
					flops = dlperf.FlopsInformation{
						MultiplyAdds: int64(*bs[0].Flops),
					}
				} else if config.App.IsDebug {
					pp.Println("cannot get flops for " + bs[0].Name + " using builtin flops computation")
				}
			}

			if len(bs) > 0 {
				totalTime = totalTime + bs[0].RealTime
				totalFlops = totalFlops.Add(flops)
			}
			if !benchInfoShort {
				// generate latex table output
				// fmt.Println("\\texttt{"+strings.ReplaceAll(lyr.Name(), "_", "\\_")+"}", " & ", lyr.OperatorType(),
				// " & ", float64(bs[0].RealTime.Nanoseconds())/float64(time.Microsecond), "&", utils.Flops(uint64(flops.Total())), " \\\\")
				return &bench{
					Type:      ty,
					Benchmark: bs[0],
					Flops:     flops,
					Layer:     lyr,
				}
			}
			return nil
		}

		switch strings.ToLower(lyr.OperatorType()) {
		case "relu", "pooling", "softmax", "dropout", "gemm", "matmul", "elementwise":
			benches := []*bench{}
			filter := filterBenchmarks(false, benchInfoDataType, "")
			bs, err := getBenchmarkTime(filter)
			if err != nil {
				continue
			}
			benches = append(benches, makeLayerInfos(bs, "forward"))

			if benchInfoTraining {
				filter := filterBenchmarks(true, benchInfoDataType, "")
				bs, err := getBenchmarkTime(filter)
				if err != nil {
					continue
				}
				benches = append(benches, makeLayerInfos(bs, "backward"))
			}

			benchmarkInfo = append(benchmarkInfo,
				benchmarkGraphNode{
					GraphNode:  nd,
					Benchmarks: benches,
				},
			)

		case "conv":
			benches := []*bench{}
			l := lyr.(*perflayer.Conv)

			filter := filterBenchmarks(false, benchInfoDataType, "")
			bs, err := getBenchmarkTime(filter)
			if err != nil {
				continue
			}
			benches = append(benches, makeLayerInfos(bs, "forward"))

			if l.HasBias() {
				filter := filterBenchmarks(false, benchInfoDataType, "", dlperf.FwdBenchmarkArgsOption.ConvFwdType(dlperf.ConvFwdTypeBias))
				bs, err := getBenchmarkTime(filter)
				if err != nil {
					continue
				}
				benches = append(benches, makeLayerInfos(bs, "bias"))
			}

			if benchInfoTraining {
				filter := filterBenchmarks(true, benchInfoDataType, "", dlperf.BwdBenchmarkArgsOption.ConvBwdType(dlperf.ConvBwdTypeData))
				bs, err := getBenchmarkTime(filter)
				if err != nil {
					pp.Println("unable to get conv data")
					continue
				}
				benches = append(benches, makeLayerInfos(bs, "backward_data"))

				filter = filterBenchmarks(true, benchInfoDataType, "", dlperf.BwdBenchmarkArgsOption.ConvBwdType(dlperf.ConvBwdTypeFilter))
				bs, err = getBenchmarkTime(filter)
				if err != nil {
					pp.Println("unable to get conv filter because of " + err.Error())
					continue
				}
				benches = append(benches, makeLayerInfos(bs, "backward_filter"))

				if l.HasBias() {
					filter = filterBenchmarks(true, benchInfoDataType, "", dlperf.BwdBenchmarkArgsOption.ConvBwdType(dlperf.ConvBwdTypeBias))
					bs, err = getBenchmarkTime(filter)
					if err != nil {
						pp.Println("unable to get conv bias")
						continue
					}
					benches = append(benches, makeLayerInfos(bs, "backward_bias"))
				}
			}

			benchmarkInfo = append(benchmarkInfo,
				benchmarkGraphNode{
					GraphNode:  nd,
					Benchmarks: benches,
				},
			)

		case "batchnorm":
			benches := []*bench{}
			var filter *benchmark.Benchmark
			if benchInfoTraining {
				filter = filterBenchmarks(false, benchInfoDataType, "", dlperf.FwdBenchmarkArgsOption.IsTraining(true))
			} else {
				filter = filterBenchmarks(false, benchInfoDataType, "", dlperf.FwdBenchmarkArgsOption.IsTraining(false))
			}
			bs, err := getBenchmarkTime(filter)
			if err != nil {
				continue
			}
			benches = append(benches, makeLayerInfos(bs, "forward"))

			if benchInfoTraining {
				filter := filterBenchmarks(true, benchInfoDataType, "")
				bs, err := getBenchmarkTime(filter)
				if err != nil {
					continue
				}
				benches = append(benches, makeLayerInfos(bs, "backward"))
			}

			benchmarkInfo = append(benchmarkInfo,
				benchmarkGraphNode{
					GraphNode:  nd,
					Benchmarks: benches,
				},
			)

		default:
			pp.Println(lyr.OperatorType())

		}
	}

	if benchSuite.GPUInformation != nil && len(benchSuite.GPUInformation.GPUS) != 0 {
		for _, gpu := range benchSuite.GPUInformation.GPUS {
			fmt.Println(gpu.ProductName)
		}
	}

	if traversalStrategy == "parallel" {
		// we need to build a new graph with
		// the benchmark times as weights
		timeTransformFunction := func(d time.Duration) float64 {
			sec := d / time.Millisecond
			nsec := d % time.Millisecond
			ms := float64(sec) + float64(nsec)/float64(time.Millisecond)
			if ms == 0 {
				return 0
			}
			return 1.0 / ms
		}
		grph := makeBenchmarkGraph(model, benchmarkInfo, timeTransformFunction)

		var firstBenchmark, lastBenchmark *benchmarkGraphNode
		for ii := range benchmarkInfo {
			bench := benchmarkInfo[ii]
			if len(bench.Benchmarks) == 0 {
				continue
			}
			if firstBenchmark == nil {
				firstBenchmark = &bench
			}
			lastBenchmark = &bench
		}

		if true {
			shortestPath := path.DijkstraFrom(firstBenchmark, grph)
			path, weight := shortestPath.To(lastBenchmark.ID())
			pp.Println(firstBenchmark.Layer().Name())
			pp.Println(lastBenchmark.Layer().Name())
			pp.Println(weight)

			pp.Println(path)
			for _, s := range path {
				g := s.(*onnx.GraphNode)
				pp.Println(g.Name)
			}
		}

		if showBenchInfo {
			dotEnc, err := dot.Marshal(grph, model.GetName(), "", "  ")
			if err != nil {
				return err
			}

			img, err := dotToImage(dotEnc)
			if err != nil {
				return err
			}

			println(img)
		}

		return nil

	}

	benchmarkInfo = append(benchmarkInfo,
		benchmarkGraphNode{
			Benchmarks: []*bench{
				&bench{
					Benchmark: benchmark.Benchmark{
						Name:     "Total",
						RealTime: totalTime,
					},
					Flops: totalFlops,
				},
			},
		})

	writer := NewWriter(bench{}, humanFlops)
	defer writer.Close()

	for _, lyr := range benchmarkInfo {
		for _, bench := range lyr.Benchmarks {
			writer.Row(bench)
		}
	}

	return nil
}

// benchinfoCmd represents the benchinfo command
var benchinfoCmd = &cobra.Command{
	Use:     "benchinfo",
	Aliases: []string{"info", "recall"},
	Short:   "Prints out the benchmark information",
	PreRunE: func(cmd *cobra.Command, args []string) error {
		traversalStrategy = strings.TrimSpace(strings.ToLower(traversalStrategy))
		if traversalStrategy != "parallel" && traversalStrategy != "serial" {
			return errors.New("invalid traversal strategy can be either `parallel` or `serial`")
		}
		return nil
	},
	RunE: benchinfo,
}

func init() {
	benchmarkResultsFolder = filepath.Join(sourcepath.MustAbsoluteDir(), "..", "results")
	benchinfoCmd.PersistentFlags().BoolVar(&benchInfoTraining, "training", false, "compute the training information")
	benchinfoCmd.PersistentFlags().BoolVar(&benchInfoShort, "short", false, "only get info about the total, rather than reporting per-layer information")
	benchinfoCmd.PersistentFlags().StringVar(&benchInfoDataType, "datatype", "float32", "compute the information for the specified scalar datatype")
	benchinfoCmd.PersistentFlags().StringVar(&benchmarkResultsFolder, "benchmark_database", benchmarkResultsFolder, "path to the benchmark results folder")
	benchinfoCmd.PersistentFlags().StringVar(&traversalStrategy, "strategy", defaultTraversalStrategy, "strategy to traverse the graph either can be `parallel` which would find the shortest path or `serial` to get the total time as if each layer is executed serially")
	benchinfoCmd.PersistentFlags().BoolVar(&showBenchInfo, "show", false, "generate the benchmark info graph (only for parallel for now)")
	rootCmd.AddCommand(benchinfoCmd)
}
