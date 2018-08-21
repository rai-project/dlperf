package dlperf

func (lyrs Layers) FwdUnion(datatype, algorithm string) Layers {
	res := []Layer{}
outer:
	for ii, lyrI := range lyrs {
		if lyrI == nil {
			continue
		}
		bI := lyrI.FwdBenchmarkFilter(datatype, algorithm)
		for jj := ii + 1; jj < len(lyrs); jj++ {
			lyrJ := lyrs[jj]
			if lyrJ == nil {
				continue
			}
			bJ := lyrJ.FwdBenchmarkFilter(datatype, algorithm)
			if bI.IsEqual(bJ) {
				continue outer
			}
		}
		res = append(res, lyrI)
	}
	return res
}
