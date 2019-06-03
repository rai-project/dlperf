package communication

type Accelerator int
type System int

const (
	UnknownAccelerator Accelerator = 0
	CPUAccelerator     Accelerator = 1
	GPUAccelerator     Accelerator = 2
)

const (
	UnknownSystem        System = 0
	SystemIBMP8          System = 0
	SystemIBMP9          System = 1
	SystemSuperMicro4096 System = 3
)

var (
	Accelerator = []Accelerator{
		CPUAccelerator,
		GPUAccelerator,
	}
	Systems = []System{
		SystemIBMP8,
		SystemIBMP9,
		SystemSuperMicro4096,
	}
)
