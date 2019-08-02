//go:generate go get -v github.com/jteeuwen/go-bindata/...
//go:generate go get -v github.com/elazarl/go-bindata-assetfs/...
//go:generate go-bindata -nomemcopy -prefix results/ -prefix assets/ -pkg cmd -o cmd/results_static.go -ignore=.DS_Store  -ignore=README.md results/... assets/...
//go:generate go generate github.com/rai-project/dlperf/pkg
//go:generate go generate github.com/rai-project/dlperf/pkg/layer
//go:generate go generate github.com/rai-project/dlperf/pkg/communication
//go:generate go generate github.com/rai-project/dlperf/pkg/benchmark
//go:generate go generate github.com/rai-project/dlperf/pkg/cloud_cost

package main
