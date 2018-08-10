// +build ignore

package main

import (
	"fmt"
	"os"

	"github.com/rai-project/dlperf/cmd"
)

func main() {
	if err := cmd.Execute(); err != nil {
		fmt.Println(err)
		os.Exit(-1)
	}
}
