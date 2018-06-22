package layer

import "encoding/json"

type Base struct {
	name string `json:"name,omitempty"`
}

func (c Base) Name() string {
	return c.name
}

func (c *Base) SetName(s string) {
	c.name = s
}

func (d *Base) UnmarshalJSON(b []byte) error {
	return json.Unmarshal(b, &d.name)
}

func (d Base) MarshalJSON() ([]byte, error) {
	return []byte(d.Name()), nil
}
