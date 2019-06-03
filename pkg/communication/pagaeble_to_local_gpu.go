package communication

import "math"

func f1(x float64) float64 {
	return 0.03512047702508423 + 1/(0.06649350465672359+20424.329033307757*math.Exp(-0.7606057225960594*x))
}
