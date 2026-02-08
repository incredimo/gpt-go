package pkg

import (
	"math"

	"github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/variable"
)

// PrecomputeFreqsCis returns cos and sin matrices for RoPE.
// dim: embedding dimension (head_size)
// maxSeqLen: maximum sequence length
// Returns: cos, sin matrices of shape (maxSeqLen, dim)
func PrecomputeFreqsCis(dim, maxSeqLen int) (*variable.Variable, *variable.Variable) {
	freqs := make([]float64, dim/2)
	for i := 0; i < dim/2; i++ {
		freqs[i] = 1.0 / math.Pow(10000.0, float64(2*i)/float64(dim))
	}

	emb := make([][]float64, maxSeqLen)
	for t := 0; t < maxSeqLen; t++ {
		emb[t] = make([]float64, dim)
		for i := 0; i < dim/2; i++ {
			val := float64(t) * freqs[i]
			emb[t][i] = val
			emb[t][i+dim/2] = val
		}
	}

	embVar := variable.NewOf(emb...) // (maxSeqLen, dim)

	// We treat these as constants (no gradient required usually, but autograd tracks everything)
	// If we don't want gradients flowing into frequencies (we don't), we can leave them as is.
	// Since they are created from scratch, they are leaves.

	cos := function.Cos(embVar)
	sin := function.Sin(embVar)

	return cos, sin
}

// ApplyRotaryEmb applies RoPE rotation.
// x: (T, D)
// cos, sin: (T, D) - must match x length
func ApplyRotaryEmb(x, cos, sin *variable.Variable) *variable.Variable {
	xRot := RotateHalf(x)

	// x * cos + rotate_half(x) * sin
	// Use variable.Mul (element-wise) and variable.Add
	return variable.Add(variable.Mul(x, cos), variable.Mul(xRot, sin))
}

func RotateHalf(x *variable.Variable) *variable.Variable {
	x1, x2 := SplitHalf(x)
	negX2 := variable.MulC(-1.0, x2)
	return Cat(negX2, x1)
}

func SplitHalf(x *variable.Variable) (*variable.Variable, *variable.Variable) {
	d := len(x.Data[0])
	half := d / 2

	t := variable.Transpose(x) // (D, T)

	idx1 := make([]float64, half)
	for i := 0; i < half; i++ {
		idx1[i] = float64(i)
	}

	idx2 := make([]float64, half)
	for i := 0; i < half; i++ {
		idx2[i] = float64(half + i)
	}

	x1t := Rows(t, idx1...)
	x2t := Rows(t, idx2...)

	x1 := variable.Transpose(x1t)
	x2 := variable.Transpose(x2t)

	return x1, x2
}
