package pkg

import (
	"math"
	"math/rand/v2"
	"sort"

	"github.com/itsubaki/autograd/matrix"
	"github.com/itsubaki/autograd/variable"
)

var (
	Add   = variable.Add
	Div   = variable.Div
	Zeros = variable.Zero
)

// Sample returns a random index based on the given probabilities.
func Sample(probs *variable.Variable) float64 {
	r := rand.Float64()

	// Find the first index where cumulative probability exceeds r.
	cumulativeProb := 0.0
	for i, p := range probs.Data[0] {
		cumulativeProb += p
		if r < cumulativeProb {
			return float64(i)
		}
	}

	return float64(len(probs.Data)) - 1
}

// SampleTemp returns a random index based on the given probabilities and temperature.
// The higher the temperature, the more random the sampling.
// Usually, temperature is between 0.5 and 0.8.
func SampleTemp(probs *variable.Variable, temperature float64) float64 {
	if temperature == 1.0 {
		return Sample(variable.NewOf(probs.Data[0]))
	}

	adjustedProbs := make([]float64, len(probs.Data[0]))
	copy(adjustedProbs, probs.Data[0])

	// Lower temperature: higher probs amplified, lower reduced, more deterministic.
	// Higher temperature: probabilities become more uniform, more random.
	sum := 0.0
	for i, p := range adjustedProbs {
		// Apply temperature by raising to power of 1/temperature.
		adjustedProbs[i] = math.Pow(p, 1.0/temperature)
		sum += adjustedProbs[i]
	}

	for i := range adjustedProbs {
		adjustedProbs[i] /= sum
	}

	return Sample(variable.NewOf(adjustedProbs))
}

// SampleAdvanced applies top-k, top-p (nucleus), and repetition penalty
// before temperature-scaled sampling. This dramatically improves perceived
// generation quality without any model changes.
//
//   - topK: keep only the top-K highest probability tokens (0 = disabled)
//   - topP: keep smallest set of tokens whose cumulative probability >= topP (1.0 = disabled)
//   - context: recent token IDs for repetition penalty
//   - repPenalty: factor to reduce probability of recent tokens (1.0 = disabled)
func SampleAdvanced(probs *variable.Variable, temperature float64, topK int, topP float64, context []float64, repPenalty float64) float64 {
	p := make([]float64, len(probs.Data[0]))
	copy(p, probs.Data[0])

	// 1. Repetition penalty: reduce probability of recently seen tokens.
	if repPenalty > 1.0 && len(context) > 0 {
		p = applyRepetitionPenalty(p, context, repPenalty)
	}

	// 2. Top-K: zero out everything outside the top-K most probable tokens.
	if topK > 0 && topK < len(p) {
		p = applyTopK(p, topK)
	}

	// 3. Top-P (nucleus): keep the smallest set of tokens summing to >= topP.
	if topP > 0.0 && topP < 1.0 {
		p = applyTopP(p, topP)
	}

	return SampleTemp(variable.NewOf(p), temperature)
}

// applyTopK keeps only the top-K highest probability tokens and zeros the rest.
func applyTopK(probs []float64, k int) []float64 {
	// Find the k-th largest probability
	sorted := make([]float64, len(probs))
	copy(sorted, probs)
	sort.Float64s(sorted)
	threshold := sorted[len(sorted)-k]

	result := make([]float64, len(probs))
	sum := 0.0
	for i, p := range probs {
		if p >= threshold {
			result[i] = p
			sum += p
		}
	}

	// Renormalize
	if sum > 0 {
		for i := range result {
			result[i] /= sum
		}
	}
	return result
}

// applyTopP (nucleus sampling) keeps the smallest set of tokens whose
// cumulative probability is >= p. This adapts the number of candidates
// dynamically — confident predictions use fewer tokens, uncertain ones more.
func applyTopP(probs []float64, p float64) []float64 {
	type indexedProb struct {
		index int
		prob  float64
	}

	// Sort by probability descending
	indexed := make([]indexedProb, len(probs))
	for i, prob := range probs {
		indexed[i] = indexedProb{i, prob}
	}
	sort.Slice(indexed, func(i, j int) bool {
		return indexed[i].prob > indexed[j].prob
	})

	// Keep tokens until cumulative probability reaches p
	result := make([]float64, len(probs))
	cumulative := 0.0
	for _, ip := range indexed {
		result[ip.index] = ip.prob
		cumulative += ip.prob
		if cumulative >= p {
			break
		}
	}

	// Renormalize
	sum := 0.0
	for _, v := range result {
		sum += v
	}
	if sum > 0 {
		for i := range result {
			result[i] /= sum
		}
	}
	return result
}

// applyRepetitionPenalty reduces the probability of tokens that appear in
// the recent context. This prevents the model from getting stuck in loops.
func applyRepetitionPenalty(probs []float64, recentTokens []float64, penalty float64) []float64 {
	result := make([]float64, len(probs))
	copy(result, probs)

	// Build set of recently seen token IDs
	seen := make(map[int]bool)
	for _, tok := range recentTokens {
		seen[int(tok)] = true
	}

	// Reduce probability of seen tokens
	sum := 0.0
	for i := range result {
		if seen[i] {
			result[i] /= penalty
		}
		sum += result[i]
	}

	// Renormalize
	if sum > 0 {
		for i := range result {
			result[i] /= sum
		}
	}
	return result
}

// Returns rows at specified indexes. Negative indexes return rows from the end.
func Rows(x *variable.Variable, indexes ...float64) *variable.Variable {
	size := len(x.Data)

	var intIndexes []int
	for _, index := range indexes {
		intIndex := int(index)
		if intIndex < 0 {
			intIndex = size + intIndex
		}

		intIndexes = append(intIndexes, intIndex)
	}

	return (&variable.Function{Forwarder: &variable.GetItemT{Slices: intIndexes}}).First(x)
}

// Returns a matrix of random values from a normal distribution.
func Normal(rows, cols int) *variable.Variable {
	rnd := func(_ float64) float64 {
		// Standard deviation = 0.02 is widely used in transformer models like GPT-2.
		// It prevents too large values in the beginning of training.
		std := 0.02
		return rand.NormFloat64() * std
	}

	m := matrix.Zero(rows, cols)
	m = matrix.F(m, rnd)

	return variable.NewOf(m...)
}

func Tril(m *variable.Variable) *variable.Variable {
	result := variable.ZeroLike(m)
	for i := 0; i < len(m.Data); i++ {
		for j := 0; j < len(m.Data[i]); j++ {
			if j <= i {
				result.Data[i][j] = m.Data[i][j]
			}
		}
	}

	return result
}

// The result would be added to computation graph and tied to m.
func MaskedInfFill(m, mask *variable.Variable) *variable.Variable {
	negInfMaskedData := matrix.F2(m.Data, mask.Data, func(a, b float64) float64 {
		if b == 0 {
			return math.Inf(-1)
		}

		return a
	})
	mMasked := Add(variable.Mul(m, mask), variable.NewOf(negInfMaskedData...))

	return mMasked
}

func DivC(c float64, x *variable.Variable) *variable.Variable {
	return variable.MulC(1.0/c, x)
}

// Returns a matrix of ones.
func Ones(m, n int) *variable.Variable {
	out := make([][]float64, m)
	for i := range m {
		out[i] = make([]float64, n)
		for j := range n {
			out[i][j] = 1.0
		}
	}

	return variable.NewOf(out...)
}

// Returns the first element of the variable.
func Val(x *variable.Variable) float64 {
	return x.Data[0][0]
}

func Flat(x *variable.Variable) []float64 {
	return matrix.Flatten(x.Data)
}

func Millions(num int) float64 {
	return float64(num) / 1e6
}

func DisableDropout() {
	variable.Config.Train = false // disables dropout
}
