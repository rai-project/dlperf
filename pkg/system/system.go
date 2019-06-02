package system

const (
	Gbps = float64(1)
	GBps = float64(8)
)

type NumaMachine struct {
	CPU []CPU
	GPU []GPU
}

type Machine struct {
	Numa []NumaMachine
}

type CPU struct {
	Interconnect Interconnect
}

type GPU struct {
	Interconnect Interconnect
}

type Interconnect interface {
	Name() string
	Bandwidth() float64
}

type System struct {
	Machines []Machine
	Network  Interconnect
}
