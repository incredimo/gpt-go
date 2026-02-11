package pkg

import (
	"math"

	"github.com/itsubaki/autograd/variable"
)

// GELU (Gaussian Error Linear Unit) approximation using the tanh formulation.
// GELU provides a smoother optimization landscape than ReLU and is the standard
// activation for modern LLMs (GPT-2, GPT-3, Llama, etc).
//
// Formula: 0.5 * x * (1 + Tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
//
// Implemented using autograd primitives so backpropagation works automatically.
func GELU(x *variable.Variable) *variable.Variable {
	c1 := 0.044715
	c2 := math.Sqrt(2.0 / math.Pi)

	// x^3
	pow3 := variable.Pow(3)(x)

	// x + 0.044715 * x^3
	inner := variable.Add(x, variable.MulC(c1, pow3))

	// Tanh(sqrt(2/pi) * (x + 0.044715 * x^3))
	tanh := variable.Tanh(variable.MulC(c2, inner))

	// 0.5 * x * (1 + Tanh(...))
	out := variable.Mul(
		variable.MulC(0.5, x),
		variable.AddC(1.0, tanh),
	)

	return out
}
