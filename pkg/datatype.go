package dlperf

type DataType struct {
	Name  string `json:"name"`
	CType string `json:"ctype"`
}

var (
	DataTypeInt8 = DataType{
		Name:  "Int8",
		CType: "int8_t",
	}
	DataTypeInt16 = DataType{
		Name:  "Int16",
		CType: "int16_t",
	}
	DataTypeInt32 = DataType{
		Name:  "Int32",
		CType: "int32_t",
	}
	DataTypeFloat16 = DataType{
		Name:  "Float16",
		CType: "__half",
	}
	DataTypeTensorCoreHalf = DataType{
		Name:  "TensorCoreHalf",
		CType: "__half",
	}
	DataTypeFloat32 = DataType{
		Name:  "Float32",
		CType: "float",
	}
	DataTypeFloat64 = DataType{
		Name:  "Float64",
		CType: "double",
	}
)

var AllDataTypes = []DataType{
	// DataTypeInt8,
	// DataTypeInt16,
	// DataTypeInt32,
	// DataTypeFloat16,
	// DataTypeTensorCoreHalf,
	DataTypeFloat32,
	// DataTypeFloat64,
}
