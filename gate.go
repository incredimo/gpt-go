package main

import (
	"github.com/itsubaki/autograd/layer"
	"github.com/itsubaki/autograd/variable"
)

// ExitGate is the "Am I done thinking?" neuron. It inspects the current
// thought vector (hidden state) and outputs a halt probability:
//   - 0.0 → "keep thinking, I need more reasoning loops"
//   - 1.0 → "I'm confident, ready to speak"
//
// This is the core mechanism for Latent Recurrence / Adaptive Compute.
// Easy tokens (like "the") trigger immediate exit. Hard tokens (like the
// answer to a riddle) force the model to loop multiple times, making the
// network effectively deeper for that specific moment.
type ExitGate struct {
	linear *Linear
}

// NewExitGate creates a gate that projects from embedSize → 1 → Sigmoid.
func NewExitGate(embedSize int) *ExitGate {
	return &ExitGate{
		linear: NewLinear(embedSize, 1),
	}
}

// Forward computes the halt probability for each token position.
// Input: (blockSize, embedSize) → Output: (blockSize, 1) with values in [0, 1].
func (g *ExitGate) Forward(input *variable.Variable) *variable.Variable {
	logits := g.linear.Forward(input)
	return Sigmoid(logits)
}

func (g *ExitGate) Params() []layer.Parameter {
	return g.linear.Params()
}
