package dlperf

import (
	"context"
	"sync"
	"runtime"

	"golang.org/x/sync/errgroup"
)

func (lyrs Layers) FwdUnion(datatype, algorithm string) Layers {
	res := []Layer{}
	var mut sync.Mutex

	g, _ := errgroup.WithContext(context.Background())

	concurrencyLimit := runtime.NumCPU()
	sem := make(chan bool, concurrencyLimit)

	for ii0 := range lyrs {
		ii := ii0
		lyrI := lyrs[ii]
		if lyrI == nil {
			continue
		}

		sem <- true

		g.Go(func() error {
				defer func() { <-sem }()
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
	for i := 0; i < cap(sem); i++ {
		sem <- true
	}
	if err := g.Wait(); err != nil {
		log.Fatal(err)
		return nil
	}
	return res
}
