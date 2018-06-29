package layer

import "encoding/json"

type Base struct {
	name              string    `json:"name,omitempty"`
	InputsDimensions  [][]int64 `json:"inputs_dimensions,omitempty"`
	OutputsDimensions [][]int64 `json:"outputs_dimensions,omitempty"`
}

func (b Base) Name() string {
	return b.name
}

func (b *Base) SetName(s string) {
	b.name = s
}

func (b *Base) UnmarshalJSON(d []byte) error {
	return json.Unmarshal(d, &b.name)
}

func (b Base) MarshalJSON() ([]byte, error) {
	return []byte(b.Name()), nil
}
