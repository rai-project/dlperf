// Code generated by "enumer -type=ConvFwdType -json -text -yaml -sql"; DO NOT EDIT.

package dlperf

import (
	"database/sql/driver"
	"encoding/json"
	"fmt"
)

const _ConvFwdTypeName = "ConvFwdTypeUndefinedConvFwdTypeConvConvFwdTypeBiasConvFwdTypeConvFusedActivation"

var _ConvFwdTypeIndex = [...]uint8{0, 20, 35, 50, 80}

func (i ConvFwdType) String() string {
	if i < 0 || i >= ConvFwdType(len(_ConvFwdTypeIndex)-1) {
		return fmt.Sprintf("ConvFwdType(%d)", i)
	}
	return _ConvFwdTypeName[_ConvFwdTypeIndex[i]:_ConvFwdTypeIndex[i+1]]
}

var _ConvFwdTypeValues = []ConvFwdType{0, 1, 2, 3}

var _ConvFwdTypeNameToValueMap = map[string]ConvFwdType{
	_ConvFwdTypeName[0:20]:  0,
	_ConvFwdTypeName[20:35]: 1,
	_ConvFwdTypeName[35:50]: 2,
	_ConvFwdTypeName[50:80]: 3,
}

// ConvFwdTypeString retrieves an enum value from the enum constants string name.
// Throws an error if the param is not part of the enum.
func ConvFwdTypeString(s string) (ConvFwdType, error) {
	if val, ok := _ConvFwdTypeNameToValueMap[s]; ok {
		return val, nil
	}
	return 0, fmt.Errorf("%s does not belong to ConvFwdType values", s)
}

// ConvFwdTypeValues returns all values of the enum
func ConvFwdTypeValues() []ConvFwdType {
	return _ConvFwdTypeValues
}

// IsAConvFwdType returns "true" if the value is listed in the enum definition. "false" otherwise
func (i ConvFwdType) IsAConvFwdType() bool {
	for _, v := range _ConvFwdTypeValues {
		if i == v {
			return true
		}
	}
	return false
}

// MarshalJSON implements the json.Marshaler interface for ConvFwdType
func (i ConvFwdType) MarshalJSON() ([]byte, error) {
	return json.Marshal(i.String())
}

// UnmarshalJSON implements the json.Unmarshaler interface for ConvFwdType
func (i *ConvFwdType) UnmarshalJSON(data []byte) error {
	var s string
	if err := json.Unmarshal(data, &s); err != nil {
		return fmt.Errorf("ConvFwdType should be a string, got %s", data)
	}

	var err error
	*i, err = ConvFwdTypeString(s)
	return err
}

// MarshalText implements the encoding.TextMarshaler interface for ConvFwdType
func (i ConvFwdType) MarshalText() ([]byte, error) {
	return []byte(i.String()), nil
}

// UnmarshalText implements the encoding.TextUnmarshaler interface for ConvFwdType
func (i *ConvFwdType) UnmarshalText(text []byte) error {
	var err error
	*i, err = ConvFwdTypeString(string(text))
	return err
}

// MarshalYAML implements a YAML Marshaler for ConvFwdType
func (i ConvFwdType) MarshalYAML() (interface{}, error) {
	return i.String(), nil
}

// UnmarshalYAML implements a YAML Unmarshaler for ConvFwdType
func (i *ConvFwdType) UnmarshalYAML(unmarshal func(interface{}) error) error {
	var s string
	if err := unmarshal(&s); err != nil {
		return err
	}

	var err error
	*i, err = ConvFwdTypeString(s)
	return err
}

func (i ConvFwdType) Value() (driver.Value, error) {
	return i.String(), nil
}

func (i *ConvFwdType) Scan(value interface{}) error {
	if value == nil {
		return nil
	}

	str, ok := value.(string)
	if !ok {
		bytes, ok := value.([]byte)
		if !ok {
			return fmt.Errorf("value is not a byte slice")
		}

		str = string(bytes[:])
	}

	val, err := ConvFwdTypeString(str)
	if err != nil {
		return err
	}

	*i = val
	return nil
}
