package main

import (
	"github.com/rai-project/config"
	"github.com/rai-project/logger"
	"github.com/sirupsen/logrus"
)

var (
	log *logrus.Entry = logger.New().WithField("pkg", "dlperf")
)

func init() {
	config.AfterInit(func() {
		log = logger.New().WithField("pkg", "dlperf")
	})
}
