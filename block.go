package main

import (
	"math"

	"github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/layer"
	"github.com/itsubaki/autograd/variable"

	"github.com/zakirullin/gpt-go/pkg"
)

var (
	Zeros               = variable.Zero
	Ones                = pkg.Ones
	GELU                = pkg.GELU
	Sigmoid             = function.Sigmoid
	Dropout             = function.DropoutSimple
	MatMul              = pkg.MatMul
	Add                 = variable.Add
	Sub                 = variable.Sub
	MulC                = variable.MulC
	Transpose           = variable.Transpose
	Softmax             = function.Softmax
	SoftmaxCrossEntropy = function.SoftmaxCrossEntropy
	RandEmbeds          = pkg.Normal
	Rows                = pkg.Rows
	Val                 = pkg.Val
	Flat                = pkg.Flat
)

type Block struct {
	embedSize int
	headCount int
	saHead    *MultiHeadAttention
	mlp       *Linear // multi-layer perceptron
	mlpProj   *Linear // projects the output of the MLP back to the original embedding size
	norm1     *LayerNorm
	norm2     *LayerNorm
}

func NewBlock(embedSize, numHeads, layerIndex int) *Block {
	// GPT-2 style initialization scaling for residual projections.
	// Both the attention output projection AND the MLP projection are
	// residual branches and must be scaled by 1/sqrt(2 * N_layers).
	scale := 1.0 / math.Sqrt(2.0*float64(layers))

	return &Block{
		embedSize: embedSize,
		headCount: numHeads,
		saHead:    NewMultiHeadAttention(embedSize, numHeads, scale),
		mlp:       NewLinear(embedSize, embedSize*4),
		mlpProj:   NewLinear(embedSize*4, embedSize, WithScale(scale)),
		norm1:     NewLayerNorm(embedSize),
		norm2:     NewLayerNorm(embedSize),
	}
}

func (b *Block) Forward(x *variable.Variable) *variable.Variable {
	// Self-attention with Pre-Norm residual.
	// CRITICAL: residual adds to the ORIGINAL x, NOT the normalized version.
	// If you add to LN(x), you're training a different architecture.
	resid := x
	xn := b.norm1.Forward(x)
	saOut := b.saHead.Forward(xn)
	saOut = Dropout(dropout)(saOut) // Residual dropout for attention branch
	x = Add(resid, saOut)

	// Feed-forward network with Pre-Norm residual.
	resid = x
	xn = b.norm2.Forward(x)
	mlp := b.mlp.Forward(xn)
	mlp = GELU(mlp)
	mlp = b.mlpProj.Forward(mlp)
	mlp = Dropout(dropout)(mlp) // Residual dropout for MLP branch
	x = Add(resid, mlp)

	return x
}

func (b *Block) Params() []layer.Parameter {
	var params []layer.Parameter
	for _, param := range b.saHead.Params() {
		params = append(params, param)
	}
	params = append(params, b.mlp.Weight, b.mlp.Bias)
	params = append(params, b.mlpProj.Weight, b.mlpProj.Bias)
	params = append(params, b.norm1.Scale, b.norm1.Shift)
	params = append(params, b.norm2.Scale, b.norm2.Shift)

	return params
}

// NoDecayParams returns parameters that should NOT be weight-decayed:
// LayerNorm scale/shift and all biases. Decaying these hurts training.
func (b *Block) NoDecayParams() []*variable.Variable {
	params := []*variable.Variable{
		b.norm1.Scale, b.norm1.Shift,
		b.norm2.Scale, b.norm2.Shift,
	}
	if b.mlp.Bias != nil {
		params = append(params, b.mlp.Bias)
	}
	if b.mlpProj.Bias != nil {
		params = append(params, b.mlpProj.Bias)
	}
	if b.saHead.proj.Bias != nil {
		params = append(params, b.saHead.proj.Bias)
	}
	return params
}
