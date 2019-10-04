package cmd

import (
	"fmt"
	"math"
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
	"github.com/spf13/cast"
	"github.com/spf13/cobra"

	"github.com/rai-project/config"
	dlperf "github.com/rai-project/dlperf/pkg"
	"github.com/rai-project/dlperf/pkg/benchmark"
	perflayer "github.com/rai-project/dlperf/pkg/layer"
	"github.com/rai-project/dlperf/pkg/onnx"
	"github.com/rai-project/dlperf/pkg/writer"
	"gonum.org/v1/gonum/graph"
	"gonum.org/v1/gonum/graph/encoding/dot"
	"gonum.org/v1/gonum/graph/path"
)

var (
	benchmarkResultsFolder         string
	benchInfoTraining              bool
	benchInfoShort                 bool
	enableReadFlopsFromDatabase    bool
	traversalStrategy              string
	showBenchInfo                  bool
	makeGraphPlot                  bool
	showBenchInfoMetrics           bool
	highlightShortestPath          bool
	showFlopsMetricsOnly           bool
	aggregateFlops                 bool
	showKernelNamesOnly            bool
	fuseLayers                     bool = false
	trimLayerName                  bool = true
	showTotalInformation           bool = true
	metricFilterString             string
	metricFilterList               []string
	chooseCudnnHeuristicsAlgorithm bool
	defaultTraversalStrategy       = "parallel"
)

func benchinfo(cmd *cobra.Command, args []string) error {
	defer func() {
		if r := recover(); r != nil {
			fmt.Println(string(debug.Stack()))
			pp.Println("[PANIC] while computing bench information " + modelPath + " [error = " + repr.String(r) + "]")
		}
	}()

	benchInfoDataType, err := getDataTypeName(datatype)
	if err != nil {
		return err
	}

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
	dlperf.IgnoreCompareFlops = showBenchInfoMetrics

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

	totalTimeSec := float64(0)
	totalFlops := dlperf.FlopsInformation{}

	debugPrint := func(val ...interface{}) {
		if config.App.IsDebug {
			fmt.Println(val...)
		}
	}

	skipNext := 0

	for ii := range nds {
		nd := nds[ii]

		if strings.ToLower(nd.Layer().OperatorType()) == "constantinput" {
			continue
		}

		if skipNext > 0 {
			// pp.Println(strings.ToLower(nd.Layer().Name()))
			skipNext--
			continue
		}
		peekNLayer := func(offset0 int) dlperf.Layer { // skips constant input
			offset := 0
			rr := ii
			for rr < len(nds) {
				lyr := nds[rr].Layer()
				rr++
				if strings.ToLower(lyr.OperatorType()) == "constantinput" {
					continue
				}
				if offset == offset0 {
					offset = rr - 1
					break
				}
				offset++
			}
			if rr >= len(nds) {
				return nil
			}
			// pp.Println(ii)
			// pp.Println(offset0)
			// pp.Println(offset)
			return nds[offset].Layer()
		}
		peekLayer := func() dlperf.Layer {
			return peekNLayer(1)
		}
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
				// pp.Println(batchSize)
				// pp.Println(lyr.Name())
				// pp.Println(filter)
				return nil, errors.New("no benchmarks")
			}
			return bs, nil
		}

		makeLayerInfos := func(bs benchmark.Benchmarks, ty string) *bench {
			bs.Sort()

			if len(bs) == 0 {
				return nil
			}
			chosenBench := bs[0]

			isConv := func(b benchmark.Benchmark) bool {
				return strings.Contains(b.Name, "CUDNN_CONV_FWD_")
			}

			if chooseCudnnHeuristicsAlgorithm && isConv(chosenBench) {
				for _, b := range bs {
					advised, ok := b.Attributes["advised_convolution_algorithm"]
					if !ok {
						continue
					}
					used, ok := b.Attributes["convolution_algorithm"]
					if !ok {
						continue
					}
					advisedS, err := cast.ToStringE(advised)
					if err != nil {
						continue
					}
					usedS, err := cast.ToStringE(used)
					if err != nil {
						continue
					}
					if advisedS == usedS {
						chosenBench = b
						break
					}
				}
			}

			flops := lyr.Information().Flops()
			if enableReadFlopsFromDatabase {
				if chosenBench.Flops != nil && *chosenBench.Flops != -1 {
					flops = dlperf.FlopsInformation{
						MultiplyAdds: int64(*chosenBench.Flops),
					}
				} else if config.App.IsDebug {
					pp.Println("cannot get flops for " + chosenBench.Name + " using builtin flops computation")
				}
			}

			if len(bs) > 0 {
				totalTimeSec = totalTimeSec + chosenBench.RealTime.Seconds()
				totalFlops = totalFlops.Add(flops)
			}
			if !benchInfoShort {
				// generate latex table output
				// fmt.Println("\\texttt{"+strings.ReplaceAll(lyr.Name(), "_", "\\_")+"}", " & ", lyr.OperatorType(),
				// " & ", float64(bs[0].RealTime.Nanoseconds())/float64(time.Microsecond), "&", utils.Flops(uint64(flops.Total())), " \\\\")
				return &bench{
					Type:      ty,
					Benchmark: chosenBench,
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
			if info := makeLayerInfos(bs, "forward"); info != nil {
				benches = append(benches, info)
			}

			if benchInfoTraining {
				filter := filterBenchmarks(true, benchInfoDataType, "")
				bs, err := getBenchmarkTime(filter)
				if err != nil {
					continue
				}
				if info := makeLayerInfos(bs, "backward"); info != nil {
					benches = append(benches, info)
				}
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

			getForwardConv := func() error {
				filter := filterBenchmarks(
					false,
					benchInfoDataType,
					"",
					dlperf.FwdBenchmarkArgsOption.ConvFwdType(dlperf.ConvFwdTypeConv),
				)
				bs, err := getBenchmarkTime(filter)
				if err != nil {
					return err
				}
				if info := makeLayerInfos(bs, "forward"); info != nil {
					benches = append(benches, info)
				}

				if l.HasBias() {
					filter := filterBenchmarks(false, benchInfoDataType, "", dlperf.FwdBenchmarkArgsOption.ConvFwdType(dlperf.ConvFwdTypeBias))
					bs, err := getBenchmarkTime(filter)
					if err != nil {
						return err
					}
					if info := makeLayerInfos(bs, "bias"); info != nil {
						benches = append(benches, info)
					}
				}
				return nil
			}

			isActivation := func(e dlperf.Layer) bool {
				op := strings.ToLower(e.OperatorType())
				switch op {
				case "relu", "tanh", "sigmoid", "elu":
					return true
				}
				return false
			}

			isBatchNorm := func(e dlperf.Layer) bool {
				op := strings.ToLower(e.OperatorType())
				switch op {
				case "batchnorm", "bn":
					return true
				}
				return false
			}

			// check for conv -> bias -> relu pattern
			if nextLayer := peekLayer(); l.HasBias() && fuseLayers && nextLayer != nil && isActivation(nextLayer) {
				panic("to do by forcing the use of the relu activation")
				filter := filterBenchmarks(
					false,
					benchInfoDataType,
					"",
					dlperf.FwdBenchmarkArgsOption.ConvFwdType(dlperf.ConvFwdTypeConvFusedActivation),
				)
				bs, err := getBenchmarkTime(filter)
				if err != nil || len(bs) == 0 {
					err := getForwardConv()
					if err != nil {
						continue
					}
				} else {
					if info := makeLayerInfos(bs, "forward"); info != nil {
						benches = append(benches, info)
					}
				}
			} else if fuseLayers && l.HasBias() { // fuse conv+bias layer
				// panic("to do by forcing the use of the identity activation")
				// nextLayer := peekLayer()
				// pp.Println(nextLayer.OperatorType())
				filter := filterBenchmarks(
					false,
					benchInfoDataType,
					"",
					dlperf.FwdBenchmarkArgsOption.ConvFwdType(dlperf.ConvFwdTypeConvFusedActivation),
				)
				bs, err := getBenchmarkTime(filter)
				if err != nil || len(bs) == 0 {
					err := getForwardConv()
					if err != nil {
						continue
					}
				} else {
					if info := makeLayerInfos(bs, "forward"); info != nil {
						benches = append(benches, info)
					}

					if nextLayer := peekLayer(); false && fuseLayers && nextLayer != nil && isBatchNorm(nextLayer) {
						if nextLayer := peekNLayer(2); fuseLayers && nextLayer != nil && isActivation(nextLayer) {
							skipNext = 2 // we skip the next batch norm and activation since it can be computed using the fused op
						} else {
							skipNext = 1 // we skip the next batch norm since it can be computed using the fused op
						}
					} else if nextLayer := peekLayer(); fuseLayers && nextLayer != nil && isActivation(nextLayer) {
						skipNext = 1 // we skip the next activation since it can be computed using the fused op

					}
				}
			} else {
				err := getForwardConv()
				if err != nil {
					continue
				}
			}

			if benchInfoTraining {
				filter := filterBenchmarks(true, benchInfoDataType, "", dlperf.BwdBenchmarkArgsOption.ConvBwdType(dlperf.ConvBwdTypeData))
				bs, err := getBenchmarkTime(filter)
				if err != nil {
					pp.Println("unable to get conv data")
					continue
				}

				if info := makeLayerInfos(bs, "backward_data"); info != nil {
					benches = append(benches, info)
				}

				filter = filterBenchmarks(true, benchInfoDataType, "", dlperf.BwdBenchmarkArgsOption.ConvBwdType(dlperf.ConvBwdTypeFilter))
				bs, err = getBenchmarkTime(filter)
				if err != nil {
					pp.Println("unable to get conv filter because of " + err.Error())
					continue
				}

				if info := makeLayerInfos(bs, "backward_filter"); info != nil {
					benches = append(benches, info)
				}

				if l.HasBias() {
					filter = filterBenchmarks(true, benchInfoDataType, "", dlperf.BwdBenchmarkArgsOption.ConvBwdType(dlperf.ConvBwdTypeBias))
					bs, err = getBenchmarkTime(filter)
					if err != nil {
						pp.Println("unable to get conv bias")
						continue
					}

					if info := makeLayerInfos(bs, "backward_bias"); info != nil {
						benches = append(benches, info)
					}
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

			if info := makeLayerInfos(bs, "forward"); info != nil {
				benches = append(benches, info)
			}

			if benchInfoTraining {
				filter := filterBenchmarks(true, benchInfoDataType, "")
				bs, err := getBenchmarkTime(filter)
				if err != nil {
					continue
				}

				if info := makeLayerInfos(bs, "backward"); info != nil {
					benches = append(benches, info)
				}

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
			msn := d / time.Millisecond
			nsec := d % time.Millisecond
			ms := float64(msn) + float64(nsec)/float64(time.Millisecond)
			if ms == 0 {
				return 0
			}
			return -ms
		}

		timeTransformFunctionInverse := func(f float64) time.Duration {
			f = math.Abs(f)
			return time.Duration(f * float64(time.Millisecond))
		}
		grph := makeBenchmarkGraph(model, net, benchmarkInfo, timeTransformFunction)

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

		if traversalStrategy == "parallel" {
			if debugMode {
				pp.Println("firstBenchmark = " + firstBenchmark.Layer().Name() + idString(firstBenchmark))
				pp.Println("lastBenchmark = " + lastBenchmark.Layer().Name() + idString(lastBenchmark))
				for _, nd := range graph.NodesOf(grph.From(firstBenchmark.ID())) {
					pp.Println("children = " + nd.(*onnx.GraphNode).Name + idString(nd))
				}
			}

			shortestPath, _ := path.BellmanFordFrom(firstBenchmark, grph)
			if highlightShortestPath {
				path, weight := shortestPath.To(lastBenchmark.ID())

				// pp.Println(weight)

				totalTimeSec = timeTransformFunctionInverse(weight).Seconds()

				// pp.Println(totalTimeSec)

				// pp.Println(path[0])
				for ii := 1; ii < len(path); ii++ {
					grphEdge := net.Edge(path[ii-1].ID(), path[ii].ID())
					if grphEdge == nil {
						continue
					}
					onnxEdge, ok := grphEdge.(*onnx.GraphEdge)
					if !ok {
						pp.Println(grphEdge)
						continue
					}
					onnxEdge.Highlight()
				}
				_ = path
			}

			// weight := shortestPath.WeightTo(lastBenchmark.ID())
			// pp.Println(timeTransformFunctionInverse(weight))
		}

		if showBenchInfo {
			nodesToRemove := []int64{}
			for _, nd := range graph.NodesOf(grph.Nodes()) {
				to := grph.To(nd.ID())
				if to.Len() == 0 {
					nodesToRemove = append(nodesToRemove, nd.ID())
				}
			}
			for _, nd := range nodesToRemove {
				grph.RemoveNode(nd)
			}
			dotEnc, err := dot.Marshal(grph, model.GetName(), "", "  ")
			if err != nil {
				return err
			}

			img, err := dotToImage(dotEnc)
			if err != nil {
				return err
			}

			err = com.WriteFile(outputFileName+".dot", dotEnc)
			if err != nil {
				return err
			}

			println(outputFileName + ".dot")

			println(img)
		}

	}

	if showTotalInformation {
		benchmarkInfo = append(benchmarkInfo,
			benchmarkGraphNode{
				Benchmarks: []*bench{
					&bench{
						Benchmark: benchmark.Benchmark{
							Name:     "Total",
							RealTime: time.Duration(totalTimeSec * float64(time.Second)),
						},
						Flops: totalFlops,
					},
				},
			})
	}

	writer := NewWriter(
		bench{},
		writer.ShowHumanFlops(humanFlops),
		writer.ShowMetrics(showBenchInfoMetrics),
		writer.ShowFlopsMetricsOnly(showFlopsMetricsOnly),
		writer.AggregateFlopsMetrics(aggregateFlops),
		writer.HideEmptyMetrics(false),
		writer.MetricsFilter(metricFilterList),
		writer.TrimLayerName(trimLayerName),
		writer.ShowKernelNamesOnly(showKernelNamesOnly),
	)
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
		if metricFilterString != "" {
			metricFilterList = strings.Split(metricFilterString, ",")
		}
		if err := setupDLPerfDataType(datatype); err != nil {
			return err
		}
		return nil
	},
	RunE: benchinfo,
}

func init() {
	benchmarkResultsFolder = filepath.Join(sourcepath.MustAbsoluteDir(), "..", "results")
	benchinfoCmd.PersistentFlags().BoolVar(&benchInfoTraining, "training", false, "compute the training information")
	benchinfoCmd.PersistentFlags().BoolVar(&benchInfoShort, "short", false, "only get info about the total, rather than reporting per-layer information")
	benchinfoCmd.PersistentFlags().StringVar(&benchmarkResultsFolder, "benchmark_database", benchmarkResultsFolder, "path to the benchmark results folder")
	benchinfoCmd.PersistentFlags().StringVar(&traversalStrategy, "strategy", defaultTraversalStrategy, "strategy to traverse the graph either can be `parallel` which would find the shortest path or `serial` to get the total time as if each layer is executed serially")
	benchinfoCmd.PersistentFlags().BoolVar(&showBenchInfo, "show", false, "generate the benchmark info graph (only for parallel for now)")
	benchinfoCmd.PersistentFlags().BoolVar(&showBenchInfoMetrics, "metrics", false, "grabs the metrics from the benchmarks")
	benchinfoCmd.PersistentFlags().BoolVar(&makeGraphPlot, "graph", false, "generate a graphviz plot of the results")
	benchinfoCmd.PersistentFlags().BoolVar(&showKernelNamesOnly, "kernels_only", false, "show a table of only the layers and corresponding kernels")
	benchinfoCmd.PersistentFlags().BoolVar(&showFlopsMetricsOnly, "flops_only", false, "show a table of only the theoretical and actual flops for each layer")
	benchinfoCmd.PersistentFlags().BoolVar(&aggregateFlops, "flops_aggregate", false, "sum all the flops within a metric for each layer")
	benchinfoCmd.PersistentFlags().StringVar(&metricFilterString, "metric_filter", "", "filter using the command seperated list of metrics")
	benchinfoCmd.PersistentFlags().BoolVar(&showTotalInformation, "total", true, "show the total information across all layers")
	benchinfoCmd.PersistentFlags().BoolVar(&fuseLayers, "fused", false, "check for fused layers in the database")
	benchinfoCmd.PersistentFlags().BoolVar(&trimLayerName, "trim_layer_name", true, "only show the first few characters of a layer")
	benchinfoCmd.PersistentFlags().BoolVar(&chooseCudnnHeuristicsAlgorithm, "choose_cudnn_heuristics", false, "choose advised algorithm by cudnn heuristics rather than the fastest")
	benchinfoCmd.PersistentFlags().BoolVar(&highlightShortestPath, "highlight_fast_path", true, "highlight the path taken when creating the graph visualization")
	benchinfoCmd.PersistentFlags().StringVar(&datatype, "datatype", "float32", "data type to use (default is float32)")
	rootCmd.AddCommand(benchinfoCmd)
}
