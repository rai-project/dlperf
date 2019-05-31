package dlperf

import (
	"github.com/rai-project/config"
	"github.com/rai-project/logger"
	"github.com/sirupsen/logrus"
)

var (
	batchSize int64
	log       *logrus.Entry
)

func GetBatchSize() int64 {
	return batchSize
}

func SetBatchSize(size int64) {
	batchSize = size
}

func init() {
	config.AfterInit(func() {
		log = logger.New().WithField("pkg", "dlperf/pkg")
	})
}
