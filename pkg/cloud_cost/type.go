package cloud_cost

type InstanceInformation struct {
	Products []struct {
		Category      string  `json:"category"`
		Type          string  `json:"type"`
		OnDemandPrice float64 `json:"onDemandPrice"`
		SpotPrice     []struct {
			Zone  string  `json:"zone"`
			Price float64 `json:"price"`
		} `json:"spotPrice"`
		CpusPerVM       int         `json:"cpusPerVm"`
		MemPerVM        int         `json:"memPerVm"`
		GpusPerVM       int         `json:"gpusPerVm"`
		NtwPerf         string      `json:"ntwPerf"`
		NtwPerfCategory string      `json:"ntwPerfCategory"`
		Zones           interface{} `json:"zones"`
		Attributes      struct {
			CPU                  string `json:"cpu"`
			InstanceTypeCategory string `json:"instanceTypeCategory"`
			Memory               string `json:"memory"`
			NetworkPerfCategory  string `json:"networkPerfCategory"`
		} `json:"attributes"`
		CurrentGen bool `json:"currentGen"`
		Burst      bool `json:"burst,omitempty"`
	} `json:"products"`
	ScrapingTime string `json:"scrapingTime"`
}
