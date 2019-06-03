package communication

func BandwidthFunc func(bytes int64) float64
func CopyLatencyFunc func(bytes int64) float64