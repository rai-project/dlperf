package communication

type BandwidthFunc func(bytes int64) float64
type CopyLatencyFunc func(bytes int64) float64
