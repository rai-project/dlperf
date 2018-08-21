package system

type NvidiaGPU struct {
	Name            string       `json:"name,omitempty"`
	Architecture    string       `json:"architecture,omitempty"`
	Interconnect    Interconnect `json:"interconnect,omitempty"`
	ClockRate       int64        `json:"clock_rate,omitempty"`
	PeekGFlops      int64        `json:"peek_gflops,omitempty"`
	MemoryBandwidth float64      `json:"memory_bandwidth,omitempty"`
}

var (
	NVIDIATeslaV100SXM2 = NvidiaGPU{
		Name:            "TESLA V100 SXM2",
		Architecture:    "Volta",
		Interconnect:    SXM2{},
		ClockRate:       int64(1530),
		PeekGFlops:      int64(15000),
		MemoryBandwidth: float64(900),
	}

	NVIDIATeslaV100PCIe = NvidiaGPU{
		Name:            "TESLA V100 PCIE",
		Architecture:    "Volta",
		Interconnect:    PCIe2{},
		ClockRate:       int64(1380),
		PeekGFlops:      int64(14000),
		MemoryBandwidth: float64(900),
	}

	NVIDIATeslaP100SXM2 = NvidiaGPU{
		Name:            "TESLA P100 SXM2",
		Architecture:    "Pascal",
		Interconnect:    SXM2{},
		ClockRate:       int64(1481),
		PeekGFlops:      int64(10600),
		MemoryBandwidth: float64(732),
	}

	NVIDIATeslaP100PCIe = NvidiaGPU{
		Name:            "TESLA P100 PCIE",
		Architecture:    "Pascal",
		Interconnect:    PCIe2{},
		ClockRate:       int64(1328),
		PeekGFlops:      int64(9300),
		MemoryBandwidth: float64(732),
	}

	NVIDIATitanXP = NvidiaGPU{
		Name:            "TITAN Xp",
		Architecture:    "Maxwell",
		Interconnect:    PCIe2{},
		ClockRate:       int64(1582),
		PeekGFlops:      int64(12100),
		MemoryBandwidth: float64(547.7),
	}

	NVIDIATitanX = NvidiaGPU{
		Name:            "TITAN X",
		Architecture:    "Maxwell",
		Interconnect:    PCIe2{},
		ClockRate:       int64(1000),
		PeekGFlops:      int64(6144),
		MemoryBandwidth: float64(336.5),
	}
	NVIDIAK20 = NvidiaGPU{
		Name:            "K20",
		Architecture:    "Kepler",
		Interconnect:    PCIe2{},
		ClockRate:       int64(1000),
		PeekGFlops:      int64(3520),
		MemoryBandwidth: float64(208),
	}
	NVIDIAK20X = NvidiaGPU{
		Name:            "K20X",
		Architecture:    "Kepler",
		Interconnect:    PCIe2{},
		ClockRate:       int64(1000),
		PeekGFlops:      int64(3935),
		MemoryBandwidth: float64(250),
	}

	NVIDIAK40 = NvidiaGPU{
		Name:            "K40",
		Architecture:    "Kepler",
		Interconnect:    PCIe2{},
		ClockRate:       int64(745),
		PeekGFlops:      int64(4290),
		MemoryBandwidth: float64(288),
	}
	NVIDIAK80 = NvidiaGPU{
		Name:            "K80",
		Architecture:    "Kepler",
		Interconnect:    PCIe2{},
		ClockRate:       int64(560),
		PeekGFlops:      int64(5600),
		MemoryBandwidth: float64(480),
	}
)
