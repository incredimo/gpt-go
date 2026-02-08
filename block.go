package main

import (
	"github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/layer"
	"github.com/itsubaki/autograd/variable"

	"github.com/zakirullin/gpt-go/pkg"
)

var (
	Zeros               = variable.Zero
	Ones                = pkg.Ones
	ReLU                = function.ReLU
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
	// SwiGLU components
	mlpGate *Linear
	mlpUp   *Linear
	mlpDown *Linear

	norm1 *RMSNorm
	norm2 *RMSNorm
}

func NewBlock(embedSize, numHeads, blockSize int) *Block {
	hiddenSize := embedSize * 4
	return &Block{
		embedSize: embedSize,
		headCount: numHeads,
		saHead:    NewMultiHeadAttention(embedSize, numHeads, blockSize),
		mlpGate:   NewLinear(embedSize, hiddenSize, NoBias()),
		mlpUp:     NewLinear(embedSize, hiddenSize, NoBias()),
		mlpDown:   NewLinear(hiddenSize, embedSize, NoBias()),
		norm1:     NewRMSNorm(embedSize),
		norm2:     NewRMSNorm(embedSize),
	}
}

func (b *Block) ClearCache() {
	b.saHead.ClearCache()
}

func (b *Block) Forward(input, cos, sin *variable.Variable, seqLen int, useCache bool) *variable.Variable {
	// Self-attention with residual connections. Input is our highway, we allow the gradient to flow back unimpeded.
	normalized := b.norm1.Forward(input)                      // Normalize input
	saOut := b.saHead.Forward(normalized, cos, sin, seqLen, useCache) // Encode relationships between positions, (blockSize, embedSize)
	input = Add(input, saOut)                                 // Add residual attention output back to main path

	// Feed-forward network with residual connection (SwiGLU)
	normalized = b.norm2.Forward(input) // Normalize input

	gate := b.mlpGate.Forward(normalized)
	up := b.mlpUp.Forward(normalized)
	act := Swish(gate)
	h := Mul(act, up)

	mlpOutput := b.mlpDown.Forward(h)
	mlpOutput = Dropout(dropout)(mlpOutput) // Dropping out some neurons to prevent overfitting
	input = Add(input, mlpOutput)           // Add feed-forward residual output to main path

	return input
}

func (b *Block) Params() []layer.Parameter {
	var params []layer.Parameter
	for _, param := range b.saHead.Params() {
		params = append(params, param)
	}
	params = append(params, b.mlpGate.Weight, b.mlpUp.Weight, b.mlpDown.Weight)
	params = append(params, b.norm1.Params()...)
	params = append(params, b.norm2.Params()...)

	return params
}
