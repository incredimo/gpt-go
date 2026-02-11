package pkg

import (
	"math"

	"github.com/itsubaki/autograd/matrix"
)

// ClipGradNorm scales all parameter gradients so their combined L2 norm
// does not exceed maxNorm. This prevents the "exploding gradient" problem
// that destabilizes deep and recurrent networks, enabling higher learning
// rates and more aggressive training schedules.
//
// Returns the total gradient norm before clipping (useful for monitoring).
func ClipGradNorm(params *Params, maxNorm float64) float64 {
	totalNormSq := 0.0

	// Calculate L2 norm of all gradients combined.
	pList := params.Params()
	for _, p := range pList {
		if p.Grad == nil {
			continue
		}
		for _, row := range p.Grad.Data {
			for _, v := range row {
				totalNormSq += v * v
			}
		}
	}

	totalNorm := math.Sqrt(totalNormSq)
	clipCoef := maxNorm / (totalNorm + 1e-6)

	// If the total norm exceeds maxNorm, scale down all gradients proportionally.
	if clipCoef < 1.0 {
		for _, p := range pList {
			if p.Grad == nil {
				continue
			}
			p.Grad.Data = matrix.MulC(clipCoef, p.Grad.Data)
		}
	}

	return totalNorm
}
