package main

import (
	"math"

	"github.com/itsubaki/autograd/layer"
	"github.com/itsubaki/autograd/variable"

	"github.com/zakirullin/gpt-go/pkg"
)

var (
	Tril          = pkg.Tril
	MaskedInfFill = pkg.MaskedInfFill
)

type MultiHeadAttention struct {
	numHeads  int
	embedSize int
	headSize  int
	Heads     []*Head
	proj      *Linear
}

// NewMultiHeadAttention creates a multi-head attention module.
// projScale applies initialization scaling to the output projection
// (residual branch), matching GPT-2's 1/sqrt(2*layers) formula.
func NewMultiHeadAttention(embedSize, numHeads int, projScale float64) *MultiHeadAttention {
	heads := make([]*Head, numHeads)
	headSize := embedSize / numHeads
	for i := range heads {
		heads[i] = NewHead(embedSize, headSize)
	}

	return &MultiHeadAttention{
		Heads:     heads,
		numHeads:  numHeads,
		embedSize: embedSize,
		headSize:  headSize,
		proj:      NewLinear(embedSize, embedSize, WithScale(projScale)),
	}
}

func (mh *MultiHeadAttention) Forward(input *variable.Variable) *variable.Variable {
	var features []*variable.Variable
	for _, head := range mh.Heads {
		features = append(features, head.Forward(input))
	}

	out := pkg.Cat(features...)
	out = mh.proj.Forward(out) // Project back to (embedSize, embedSize)
	// Dropout for the residual path is applied in Block.Forward,
	// keeping both residual dropouts in one place for clarity.

	return out
}

func (mh *MultiHeadAttention) Params() []layer.Parameter {
	var params []layer.Parameter
	for _, head := range mh.Heads {
		params = append(params, head.Query.Weight, head.Key.Weight, head.Value.Weight)
	}
	params = append(params, mh.proj.Weight, mh.proj.Bias)

	return params
}

type Head struct {
	embedSize int
	headSize  int
	Key       *Linear
	Query     *Linear
	Value     *Linear
}

func NewHead(embedSize, headSize int) *Head {
	key := NewLinear(embedSize, headSize, NoBias())
	query := NewLinear(embedSize, headSize, NoBias())
	value := NewLinear(embedSize, headSize, NoBias())

	return &Head{embedSize, headSize, key, query, value}
}

// Self-attention mechanism, see main_test.go for explanation.
func (h *Head) Forward(input *variable.Variable) *variable.Variable {
	query := h.Query.Forward(input)
	key := h.Key.Forward(input)
	attentions := MatMul(query, Transpose(key))

	T := len(input.Data) // number of tokens
	tril := Tril(Ones(T, T))
	attentions = MaskedInfFill(attentions, tril)
	attentions = Softmax(attentions)
	attentions = Dropout(dropout)(attentions) // Attention dropout (on weights, not residual)

	v := h.Value.Forward(input)
	weightedSum := MatMul(attentions, v)
	normalizedSum := MulC(math.Pow(float64(h.embedSize), -0.5), weightedSum)

	return normalizedSum
}
