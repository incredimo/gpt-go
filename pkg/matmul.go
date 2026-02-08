// MatMul performs matrix multiplication using gonum/mat for optimized BLAS performance.
package pkg

import (
	"github.com/itsubaki/autograd/matrix"
	"github.com/itsubaki/autograd/variable"
	"gonum.org/v1/gonum/mat"
)

func MatMul(x ...*variable.Variable) *variable.Variable {
	return (&variable.Function{Forwarder: &MatMulT{}}).First(x...)
}

type MatMulT struct {
	x, w *variable.Variable
}

func (f *MatMulT) Forward(x ...*variable.Variable) []*variable.Variable {
	f.x, f.w = x[0], x[1]

	m := toDense(f.x.Data)
	n := toDense(f.w.Data)

	var result mat.Dense
	result.Mul(m, n)

	rows, cols := result.Dims()
	// Use explicit [][]float64 for compatibility with variable.NewOf
	outData := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		outData[i] = make([]float64, cols)
		copy(outData[i], result.RawRowView(i))
	}

	return []*variable.Variable{variable.NewOf(outData...)}
}

func (f *MatMulT) Backward(gy ...*variable.Variable) []*variable.Variable {
	return []*variable.Variable{
		MatMul(gy[0], variable.Transpose(f.w)), // gy * w.T
		MatMul(variable.Transpose(f.x), gy[0]), // x.T * gy
	}
}

// toDense converts matrix.Matrix to *mat.Dense
func toDense(data matrix.Matrix) *mat.Dense {
	rows := len(data)
	if rows == 0 {
		return mat.NewDense(0, 0, nil)
	}
	cols := len(data[0])
	flat := make([]float64, rows*cols)
	for i := 0; i < rows; i++ {
		copy(flat[i*cols:], data[i])
	}
	return mat.NewDense(rows, cols, flat)
}
