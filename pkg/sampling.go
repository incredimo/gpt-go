package pkg

import (
	"math"
	"slices"

	"github.com/itsubaki/autograd/variable"
)

// TopK filters the logits, keeping only the top k values.
// Elements not in the top k are set to -Inf.
func TopK(logits *variable.Variable, k int) *variable.Variable {
	if k <= 0 {
		return logits
	}

	data := logits.Data[0] // Assume shape (1, vocabSize)
	n := len(data)
	if k >= n {
		return logits
	}

	// Create a copy to modify
	out := make([]float64, n)
	copy(out, data)

	// Find the k-th largest value
	// We can sort a copy of the data
	sorted := make([]float64, n)
	copy(sorted, data)
	// Sort descending
	slices.SortFunc(sorted, func(a, b float64) int {
		if a > b {
			return -1
		}
		if a < b {
			return 1
		}
		return 0
	})

	// Mask values below cutoff
	// Note: if there are duplicate values equal to cutoff, we might keep more than k.
	// This is usually acceptable. Strictly keeping k requires index tracking.
	// For "100% accuracy" I should track indices or be careful.
	// A strictly correct TopK keeps exactly k elements.
	// If multiple elements equal cutoff, we should keep them until count is k.

	// Let's do a more robust approach: Pairs of (val, index)
	type pair struct {
		val float64
		idx int
	}
	pairs := make([]pair, n)
	for i, v := range data {
		pairs[i] = pair{v, i}
	}

	slices.SortFunc(pairs, func(a, b pair) int {
		if a.val > b.val {
			return -1
		}
		if a.val < b.val {
			return 1
		}
		return 0
	})

	// Keep indices of top k
	keep := make(map[int]bool)
	for i := 0; i < k; i++ {
		keep[pairs[i].idx] = true
	}

	for i := 0; i < n; i++ {
		if !keep[i] {
			out[i] = math.Inf(-1)
		}
	}

	return variable.NewOf(out)
}

// TopP (Nucleus Sampling) filters the logits, keeping the smallest set of tokens
// whose cumulative probability exceeds p.
func TopP(logits *variable.Variable, p float64) *variable.Variable {
	if p >= 1.0 {
		return logits
	}

	data := logits.Data[0]
	n := len(data)

	// Work with probabilities for thresholding, but apply mask to logits
	// Softmax to get probs
	probs := softmax(data)

	type pair struct {
		prob float64
		idx  int
	}
	pairs := make([]pair, n)
	for i, v := range probs {
		pairs[i] = pair{v, i}
	}

	// Sort descending
	slices.SortFunc(pairs, func(a, b pair) int {
		if a.prob > b.prob {
			return -1
		}
		if a.prob < b.prob {
			return 1
		}
		return 0
	})

	cumsum := 0.0
	keep := make(map[int]bool)
	for _, pair := range pairs {
		keep[pair.idx] = true
		cumsum += pair.prob
		if cumsum > p {
			break
		}
	}

	out := make([]float64, n)
	copy(out, data)
	for i := 0; i < n; i++ {
		if !keep[i] {
			out[i] = math.Inf(-1)
		}
	}

	return variable.NewOf(out)
}

func softmax(logits []float64) []float64 {
	max := math.Inf(-1)
	for _, v := range logits {
		if v > max {
			max = v
		}
	}

	sum := 0.0
	out := make([]float64, len(logits))
	for i, v := range logits {
		out[i] = math.Exp(v - max)
		sum += out[i]
	}
	for i := range out {
		out[i] /= sum
	}
	return out
}
