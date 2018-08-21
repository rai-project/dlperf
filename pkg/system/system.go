package system

const (
	Gbps = float64(1)
	GBps = float64(8)
)

type Machine struct {
	CPU []CPU
	GPU []GPU
}

type CPU struct {
}

type GPU struct {
}

type Interconnect interface {
	Name() string
	Bandwidth() float64
}

type System struct {
	Machines []Machine
	Network  Interconnect
}
