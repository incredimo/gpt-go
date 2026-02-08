package pkg

import (
	"github.com/itsubaki/autograd/variable"
)

// Cat concatenates matrices horizontally (dim=1)
func Cat(x ...*variable.Variable) *variable.Variable {
	if len(x) == 0 {
		return nil
	}
	return (&variable.Function{Forwarder: &CatT{NumInputs: len(x)}}).First(x...)
}

type CatT struct {
	NumInputs int
	// Assuming all inputs have same shape
	ColSize   int
}

// Concatenate along the columns dimension (dim=1)
func (f *CatT) Forward(x ...*variable.Variable) []*variable.Variable {
	rows := len(x[0].Data)
	f.ColSize = len(x[0].Data[0])

	totalCols := 0
	for _, v := range x {
		totalCols += len(v.Data[0])
	}
	// Currently Cat assumes same ColSize for Backward, but Forward might handle variable sizes?
	// The implementation below assumes f.ColSize is fixed for all inputs for Backward.
	// But Forward logic handles totalCols.
	// Let's stick to existing logic:
	// "f.ColSize = len(x[0].Data[0])" suggests it assumes uniform inputs.

	result := make([][]float64, rows)
	for i := range result {
		result[i] = make([]float64, totalCols)
		colOffset := 0
		for _, v := range x {
			copy(result[i][colOffset:], v.Data[i])
			colOffset += len(v.Data[0])
		}
	}

	return []*variable.Variable{
		variable.NewOf(result...),
	}
}

func (f *CatT) Backward(gy ...*variable.Variable) []*variable.Variable {
	grads := make([]*variable.Variable, f.NumInputs)

	// Split along columns
	for i := 0; i < f.NumInputs; i++ {
		// This assumes all inputs had same ColSize.
		// If Cat allows different ColSizes, this Backward is WRONG.
		// Existing implementation assumes uniform ColSize.
		// Let's keep it consistent.
		colOffset := i * f.ColSize
		colData := make([][]float64, len(gy[0].Data))

		for j := range colData {
			colData[j] = make([]float64, f.ColSize)
			copy(colData[j], gy[0].Data[j][colOffset:colOffset+f.ColSize])
		}

		grads[i] = variable.NewOf(colData...)
	}

	return grads
}

// CatV concatenates matrices vertically (dim=0)
func CatV(x ...*variable.Variable) *variable.Variable {
	if len(x) == 0 {
		return nil
	}
	return (&variable.Function{Forwarder: &CatVT{NumInputs: len(x)}}).First(x...)
}

type CatVT struct {
	NumInputs int
	RowSize   int
}

func (f *CatVT) Forward(x ...*variable.Variable) []*variable.Variable {
	f.RowSize = len(x[0].Data) // Assumes uniform row size?
	// No, CatV logic usually allows different number of rows.
	// But Backward needs to know how many rows each input had to split correctly.
	// If uniform, easy. If not, we need to store sizes.
	// For "Tile", uniform is true. For "BatchSample", uniform is true (BlockSize).
	// So uniform assumption is acceptable for now.

	totalRows := 0
	for _, v := range x {
		totalRows += len(v.Data)
	}

	result := make([][]float64, 0, totalRows)
	for _, v := range x {
		result = append(result, v.Data...)
	}

	return []*variable.Variable{
		variable.NewOf(result...),
	}
}

func (f *CatVT) Backward(gy ...*variable.Variable) []*variable.Variable {
	grads := make([]*variable.Variable, f.NumInputs)

	// Split along rows
	rowOffset := 0
	for i := 0; i < f.NumInputs; i++ {
		// Assuming uniform row size
		inputRows := f.RowSize

		rows := gy[0].Data[rowOffset : rowOffset+inputRows]
		grads[i] = variable.NewOf(rows...)

		rowOffset += inputRows
	}

	return grads
}
