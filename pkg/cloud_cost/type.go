package cloud_cost

type SpotPrice struct {
	Zone  string  `json:"zone,omitempty"`
	Price float64 `json:"price,omitempty"`
}

type Attributes struct {
	CPU                  string `json:"cpu,omitempty"`
	InstanceTypeCategory string `json:"instanceTypeCategory,omitempty"`
	Memory               string `json:"memory,omitempty"`
	NetworkPerfCategory  string `json:"networkPerfCategory,omitempty"`
	GpuArch              string `json:"gpuArch,omitempty"`
}

type InstanceInformation struct {
	Category        string      `json:"category,omitempty"`
	Type            string      `json:"type,omitempty"`
	OnDemandPrice   float64     `json:"onDemandPrice,omitempty"`
	SpotPrice       []SpotPrice `json:"spotPrice,omitempty"`
	CpusPerVM       int         `json:"cpusPerVm,omitempty"`
	MemPerVM        int         `json:"memPerVm,omitempty"`
	GpusPerVM       int         `json:"gpusPerVm,omitempty"`
	NtwPerf         string      `json:"ntwPerf,omitempty"`
	NtwPerfCategory string      `json:"ntwPerfCategory,omitempty"`
	Zones           interface{} `json:"zones,omitempty"`
	Attributes      Attributes  `json:"attributes,omitempty"`
	CurrentGen      bool        `json:"currentGen,omitempty"`
	Burst           bool        `json:"burst,omitempty"`
}

type InstanceInformations struct {
	Products     []InstanceInformation `json:"products"`
	ScrapingTime string                `json:"scrapingTime"`
}
