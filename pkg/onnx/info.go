package onnx

import (
	"sort"

	"github.com/cevaris/ordered_map"
	"github.com/k0kubun/pp"
	"github.com/rai-project/dlperf/pkg"
	"gonum.org/v1/gonum/graph"
	"gonum.org/v1/gonum/graph/topo"
)

func sortByOrder(nds []graph.Node, layers dlperf.Layers) dlperf.Layers {

	if len(layers) == 1 {
		return layers
	}

	findNodePosition := func(name string) int {
		for ii, n0 := range nds {
			n := n0.(*GraphNode)
			if n.name == name || n.Name == name {
				return ii
			}
		}
		panic("unable to find node " + name)

		return 0
	}

	order := func(ii, jj int) bool {
		a := layers[ii]
		b := layers[jj]
		apos := findNodePosition(a.Name())
		bpos := findNodePosition(b.Name())
		return apos < bpos
	}

	sort.Slice(layers, order)

	return layers
}

func sortByDimension(layers dlperf.Layers) dlperf.Layers {
	if len(layers) == 1 {
		return layers
	}

	rest := layers[1:]

	isGreater := func(ii, jj int) bool {
		a := rest[ii]
		b := rest[jj]
		sa := a.OutputShapes()[0]
		sb := b.OutputShapes()[0]
		if len(sa) > len(sb) {
			return true
		}
		for ii := range sa {
			if sa[ii] > sb[ii] {
				return true
			}
		}
		return false
	}

	sort.Slice(rest, isGreater)

	return append(dlperf.Layers{layers[0]}, rest...)
}

func (o *Onnx) Information() ([]dlperf.LayerInformation, error) {
	ret := []dlperf.LayerInformation{}

	grph := o.ToGraph(GraphPruneInputs(false), GraphInputsAsConstantNodes(true))
	nds, err := topo.SortStabilized(grph, sortById)
	if err != nil {
		return nil, err
	}

	findNode := func(name string) *GraphNode {
		for _, n0 := range nds {
			n := n0.(*GraphNode)
			if n.name == name {
				return n
			}
		}
		panic("unable to find node " + name)

		return nil
	}

	layers := ordered_map.NewOrderedMap()

	for _, nd0 := range nds {
		nd, ok := nd0.(*GraphNode)
		if !ok {
			panic("invalid type for " + pp.Sprint(nd0))
		}
		layer := o.mkLayer(nd.NodeProto)

		if layer == nil {
			continue
		}

		layers.Set(nd.name, layer)
	}

	iter := layers.IterFunc()
	for kv, ok := iter(); ok; kv, ok = iter() {
		layer, ok := kv.Value.(dlperf.Layer)
		if !ok {
			panic("invalid layer type for " + pp.Sprint(kv.Value))
		}

		nd := findNode(kv.Key.(string))
		inputLayers := dlperf.Layers{}
		inputNodes := grph.To(nd.ID())
		for inputNodes.Next() {
			input0 := inputNodes.Node()
			input, ok := input0.(*GraphNode)
			if !ok {
				panic("invalid type for " + pp.Sprint(input0))
			}

			inputLayer, ok := layers.Get(input.name)
			if !ok {
				panic("unable to find input layer " + pp.Sprint(input))
			}

			inputLayers = append(inputLayers, inputLayer.(dlperf.Layer))
		}

		inputLayers = sortByOrder(nds, inputLayers)
		layer.SetInputs(inputLayers)

		// pp.Println(inputLayers)

		outputLayers := dlperf.Layers{}
		outputNodes := grph.From(nd.ID())
		for outputNodes.Next() {
			output0 := outputNodes.Node()
			output, ok := output0.(*GraphNode)
			if !ok {
				panic("invalid type for " + pp.Sprint(output0))
			}

			outputLayer, ok := layers.Get(output.name)
			if !ok {
				panic("unable to find input layer " + pp.Sprint(output))
			}

			outputLayers = append(outputLayers, outputLayer.(dlperf.Layer))
		}
		outputLayers = sortByOrder(nds, outputLayers)
		layer.SetOutputs(outputLayers)

		// if layer.Name() == "conv0" {
		// 	pp.Println("Infering Shape on ", layer.Name())
		// 	for _, input := range inputLayers {
		// 		pp.Println(input.Name())
		// 	}
		// }

		// defer func() {
		// 	if r := recover(); r != nil {
		// 		// pp.Println(layer.Name())
		// 		panic(r)
		// 	}

		// }()

		if len(inputLayers) > 0 {
			layer.SetInputShapes(getOutputShapes(inputLayers))
		}

		layer.InferShape(inputLayers)

		nd.layer = layer

		// pp.Println(layer.Name())
		// pp.Println(layer.OutputShapes())

		info := layer.Information()
		ret = append(ret, info)
	}

	// dotEnc, err := dot.Marshal(grph, "", "", "  ", true)
	// if err == nil {
	// 	println(string(dotEnc))
	// }

	return ret, nil
}

func (o Onnx) FlopsInformation() dlperf.FlopsInformation {
	flops := dlperf.FlopsInformation{}
	infos, err := o.Information()
	if err != nil {
		panic(err)
	}
	for _, info := range infos {
		flops = flops.Add(info.Flops())
	}
	return flops
}

func (o Onnx) MemoryInformation() dlperf.MemoryInformation {
	memory := dlperf.MemoryInformation{}
	infos, err := o.Information()
	if err != nil {
		panic(err)
	}
	for _, info := range infos {
		memory = memory.Add(info.Memory())
	}
	return memory
}

func (o Onnx) GetValueInfoDimensions(names []string) []dlperf.Shape {
	var ret []dlperf.Shape
	// for k, _ := range o.initializers {
	// 	pp.Println(k)
	// }
	for _, name := range names {
		init, ok := o.initializers[name]
		if ok {
			shp := getTensorProtoDimensions(init)
			ret = append(ret, shp)
			continue
		}

		val, ok := o.valueInfo[name]
		if ok {
			ret = append(ret, getValueInfoDimensions(val))
			continue
		}

		val, ok = o.inputs[name]
		if ok {
			ret = append(ret, getValueInfoDimensions(val))
			continue
		}

		val, ok = o.outputs[name]
		if ok {
			ret = append(ret, getValueInfoDimensions(val))
		}
	}

	return ret
}
