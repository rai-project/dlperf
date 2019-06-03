package cloud_cost

import (
	json "encoding/json"
	"strings"

	"github.com/pkg/errors"
)

var loadCache = map[string][]InstanceInformation{}

func loadGPUs(name string) ([]InstanceInformation, error) {
	infos, err := load(name)
	if err != nil {
		return nil, err
	}
	res := make([]InstanceInformation, 0, len(infos))
	for _, info := range infos {
		if info.GpusPerVM > 0 {
			res = append(res, info)
		}
	}

	return res, nil
}

func load(name string) ([]InstanceInformation, error) {
	name = strings.ToLower(name)
	if e, ok := loadCache[name]; ok && len(e) > 0 {
		return e, nil
	}

	fileName := name + "_instances.json"
	bts, err := _escFSByte(false, fileName)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to load %v", name)
	}

	var infos InstanceInformations
	err = json.Unmarshal(bts, &infos)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to unmarshal %v", name)
	}

	info := infos.Products
	loadCache[name] = info

	return info, nil
}
