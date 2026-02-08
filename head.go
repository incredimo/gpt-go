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

func NewMultiHeadAttention(embedSize, numHeads int) *MultiHeadAttention {
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
		proj:      NewLinear(embedSize, embedSize, NoBias()),
	}
}

func (mh *MultiHeadAttention) Forward(input, cos, sin *variable.Variable) *variable.Variable {
	var features []*variable.Variable
	for _, head := range mh.Heads {
		features = append(features, head.Forward(input, cos, sin))
	}

	out := pkg.Cat(features...)
	out = mh.proj.Forward(out)  // Project back to (embedSize, embedSize)
	out = Dropout(dropout)(out) // Dropping out some neurons to prevent overfitting

	return out
}

func (mh *MultiHeadAttention) Params() []layer.Parameter {
	var params []layer.Parameter
	for _, head := range mh.Heads {
		params = append(params, head.Query.Weight, head.Key.Weight, head.Value.Weight)
	}
	// proj has no bias
	params = append(params, mh.proj.Weight)

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
func (h *Head) Forward(input, cos, sin *variable.Variable) *variable.Variable {
	query := h.Query.Forward(input)
	key := h.Key.Forward(input)

	// Apply RoPE
	query = pkg.ApplyRotaryEmb(query, cos, sin)
	key = pkg.ApplyRotaryEmb(key, cos, sin)

	attentions := MatMul(query, Transpose(key))
	attentions = MulC(math.Pow(float64(h.headSize), -0.5), attentions)

	T := len(input.Data) // number of tokens
	tril := Tril(Ones(T, T))
	attentions = MaskedInfFill(attentions, tril)
	attentions = Softmax(attentions)
	attentions = Dropout(dropout)(attentions)

	v := h.Value.Forward(input)
	weightedSum := MatMul(attentions, v)

	return weightedSum
}
