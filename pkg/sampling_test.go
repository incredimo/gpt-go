package pkg

import (
	"math"
	"testing"

	"github.com/itsubaki/autograd/variable"
)

func TestTopK(t *testing.T) {
	logits := variable.NewOf([]float64{0.1, 0.5, 0.2, 0.8, 0.3})
	k := 2
	out := TopK(logits, k)

	if len(out.Data[0]) != 5 {
		t.Fatalf("expected length 5, got %d", len(out.Data[0]))
	}

	// 0.8 and 0.5 should be kept. Others should be -Inf.
	expected := []float64{math.Inf(-1), 0.5, math.Inf(-1), 0.8, math.Inf(-1)}
	for i, v := range out.Data[0] {
		if math.IsInf(expected[i], -1) {
			if !math.IsInf(v, -1) {
				t.Errorf("idx %d: expected -Inf, got %v", i, v)
			}
		} else {
			if v != expected[i] {
				t.Errorf("idx %d: expected %v, got %v", i, expected[i], v)
			}
		}
	}
}

func TestTopP(t *testing.T) {
	// probs: {0.1, 0.5, 0.2, 0.2} roughly
	// logits: ln(probs)
	logits := variable.NewOf([]float64{math.Log(0.1), math.Log(0.5), math.Log(0.2), math.Log(0.2)})
	// Sorted probs: 0.5, 0.2, 0.2, 0.1
	// Cumulative: 0.5, 0.7, 0.9, 1.0

	// p = 0.6 -> keep 0.5, 0.2 (cum=0.7 > 0.6)
	p := 0.6
	out := TopP(logits, p)

	// Expected: log(0.5) kept, log(0.2) kept (one of them or both depending on sort stability/equality), others -Inf
	// Since 0.2 appears twice, it depends.
	// 0.5 is index 1.
	// 0.2 are indices 2 and 3.
	// If stable sort or deterministic, we keep 0.5 (idx 1) and one 0.2.
	// Wait, TopP logic:
	// iterate sorted pairs.
	// pair 1: 0.5, cum=0.5. 0.5 < 0.6. Continue.
	// pair 2: 0.2, cum=0.7. 0.7 > 0.6. Break loop after adding this one?
	// Logic in TopP:
	// keep[pair.idx] = true; cumsum += pair.prob; if cumsum > p { break }
	// So yes, it includes the token that crosses the threshold.
	// So we keep 0.5 and ONE 0.2.
	// Which 0.2? slices.SortFunc is not guaranteed stable?
	// It usually is stable in Go 1.21+ but "not guaranteed".
	// Let's relax the test or make probs distinct.

	logits = variable.NewOf([]float64{math.Log(0.1), math.Log(0.6), math.Log(0.2), math.Log(0.1)})
	// Probs: 0.1, 0.6, 0.2, 0.1
	// Sorted: 0.6 (idx 1), 0.2 (idx 2), 0.1, 0.1
	// Cumsum: 0.6.
	// p = 0.5.
	// pair 1: 0.6. cum=0.6 > 0.5. Break.
	// Result: Keep only 0.6.

	out = TopP(logits, 0.5)
	expected := []float64{math.Inf(-1), math.Log(0.6), math.Inf(-1), math.Inf(-1)}

	for i, v := range out.Data[0] {
		if math.IsInf(expected[i], -1) {
			if !math.IsInf(v, -1) {
				t.Errorf("idx %d: expected -Inf, got %v", i, v)
			}
		} else {
			if math.Abs(v-expected[i]) > 1e-9 {
				t.Errorf("idx %d: expected %v, got %v", i, expected[i], v)
			}
		}
	}
}
