package dlperf

import (
	"context"
	"sync"

	"golang.org/x/sync/errgroup"
)

func (lyrs Layers) FwdUnion(datatype, algorithm string) Layers {
	res := []Layer{}
	var mut sync.Mutex

	g, _ := errgroup.WithContext(context.Background())

	for ii0 := range lyrs {
		ii := ii0
		lyrI := lyrs[ii]
		if lyrI == nil {
			continue
		}
		g.Go(func() error {
			bI := lyrI.FwdBenchmarkFilter(datatype, algorithm)
			for jj := ii + 1; jj < len(lyrs); jj++ {
				lyrJ := lyrs[jj]
				if lyrJ == nil {
					return nil
				}
				bJ := lyrJ.FwdBenchmarkFilter(datatype, algorithm)
				if bI.IsEqual(bJ) {
					return nil
				}
			}
			mut.Lock()
			defer mut.Unlock()
			res = append(res, lyrI)
			return nil
		})
	}
	if err := g.Wait(); err != nil {
		panic(err)
		return nil
	}
	return res
}
