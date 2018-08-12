package layer

import (
	"github.com/rai-project/config"
	"github.com/rai-project/logger"
	"github.com/sirupsen/logrus"
)

var (
	log *logrus.Entry = logger.New().WithField("pkg", "dlperf/layer")
)

func init() {
	config.AfterInit(func() {
		log = logger.New().WithField("pkg", "dlperf/layer")
	})
}
