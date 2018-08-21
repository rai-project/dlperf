package dlperf

func (lyrs Layers) FwdUnion(datatype, algorithm string) Layers {
	res := []Layer{}
outer:
	for ii, lyrI := range lyrs {
		bI := lyrI.FwdBenchmarkFilter(datatype, algorithm)
		for jj, lyrJ := range lyrs {
			if ii == jj {
				continue
			}
			bJ := lyrJ.FwdBenchmarkFilter(datatype, algorithm)
			if bI.IsEqual(bJ) {
				break outer
			}
		}
		res = append(res, lyrI)
	}
	return res
}
