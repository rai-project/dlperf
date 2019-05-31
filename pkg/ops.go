package dlperf

import (
	"context"
	"runtime"
	"sync"

	"github.com/rai-project/dlperf/pkg/benchmark"
	"golang.org/x/sync/errgroup"
)

func (lyrs Layers) FwdUnion(datatype, algorithm string) Layers {
	res := []Layer{}
	var mut sync.Mutex

	g, _ := errgroup.WithContext(context.Background())

	concurrencyLimit := runtime.NumCPU()
	sem := make(chan bool, concurrencyLimit)

	filterCache := map[int]benchmark.Benchmark{}
	var filterCacheLock sync.RWMutex

	getBenchmarkFilter := func(ii int) *benchmark.Benchmark {
		lyrI := lyrs[ii]
		if lyrI == nil {
			return nil
		}
		filterCacheLock.RLock()
		bI, ok := filterCache[ii]
		filterCacheLock.RUnlock()
		if !ok {
			bI = lyrI.FwdBenchmarkFilter(datatype, algorithm)
			filterCacheLock.Lock()
			filterCache[ii] = bI
			filterCacheLock.Unlock()
		}
		return &bI
	}

	for ii0 := range lyrs {
		ii := ii0
		lyrI := lyrs[ii]
		if lyrI == nil {
			continue
		}

		sem <- true

		g.Go(func() error {
			defer func() { <-sem }()
			bI := getBenchmarkFilter(ii)
			if bI == nil {
				return nil
			}
			for jj := ii + 1; jj < len(lyrs); jj++ {
				bJ := getBenchmarkFilter(jj)
				if bJ == nil {
					continue
				}
				if bI.IsEqual(*bJ) {
					return nil
				}
			}
			mut.Lock()
			res = append(res, lyrI)
			mut.Unlock()
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
