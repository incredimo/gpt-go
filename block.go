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
	// Weights are scaled by 1/sqrt(2 * N_layers) to keep variance bounded
	// at initialization, preventing signal explosion in deep networks.
	scale := 1.0 / math.Sqrt(2.0*float64(layers))

	return &Block{
		embedSize: embedSize,
		headCount: numHeads,
		saHead:    NewMultiHeadAttention(embedSize, numHeads),
		mlp:       NewLinear(embedSize, embedSize*4),
		mlpProj:   NewLinear(embedSize*4, embedSize, WithScale(scale)),
		norm1:     NewLayerNorm(embedSize),
		norm2:     NewLayerNorm(embedSize),
	}
}

func (b *Block) Forward(input *variable.Variable) *variable.Variable {
	// Self-attention with residual connections.
	input = b.norm1.Forward(input)   // Pre-Norm
	saOut := b.saHead.Forward(input) // Encode relationships
	input = Add(input, saOut)        // Residual

	// Feed-forward network with residual connection
	input = b.norm2.Forward(input)               // Pre-Norm
	mlpExpanded := b.mlp.Forward(input)          // Expand to 4x dimension
	mlpActivated := GELU(mlpExpanded)            // GELU activation (smoother than ReLU)
	mlpOutput := b.mlpProj.Forward(mlpActivated) // Project back to original dimension
	mlpOutput = Dropout(dropout)(mlpOutput)      // Dropout for regularization
	input = Add(input, mlpOutput)                // Residual

	return input
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
