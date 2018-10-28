package system

import "github.com/jaypipes/ghw"

func Topology() (*ghw.TopologyInfo, error) {
	return ghw.Topology()
}
