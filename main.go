package main

import (
	"github.com/k0kubun/pp"
	"github.com/rai-project/dlperf/cmd"
)

func main() {
	pp.WithLineInfo = true
	cmd.Execute()
}
