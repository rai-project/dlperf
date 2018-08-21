package layer

type DataType struct {
	Name  string `json:"name"`
	CType string `json:"ctype"`
}

var allDataTypes = []DataType{
	// DataType{
	// 	Name:  "Int8",
	// 	CType: "int8_t",
	// },
	// DataType{
	// 	Name:  "Int32",
	// 	CType: "int32_t",
	// },
	DataType{
		Name:  "Float16",
		CType: "__half",
	},
	DataType{
		Name:  "Float32",
		CType: "float",
	},
	// DataType{
	// 	Name:  "Float64",
	// 	CType: "double",
	// },
	DataType{
		Name:  "TensorCoreHalf",
		CType: "__half",
	},
}
