package system

type NvidiaGPU struct {
	Name            string       `json:"name,omitempty"`
	Architecture    string       `json:"architecture,omitempty"`
	Interconnect    Interconnect `json:"interconnect,omitempty"`
	ClockRate       int64        `json:"clock_rate,omitempty"`
	PeekGFlops      int64        `json:"peek_gflops,omitempty"`
	MemoryBandwidth int64        `json:"memory_bandwidth,omitempty"`
}

var (
	V100SXM2 = NvidiaGPU{
		Name:            "TESLA V100 SXM2",
		Architecture:    "V100",
		Interconnect:    SXM2{},
		ClockRate:       int64(1530),
		PeekGFlops:      int64(15000),
		MemoryBandwidth: int64(900),
	}
)
