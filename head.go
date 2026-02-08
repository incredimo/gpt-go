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
	blockSize int

	// Combined Q, K, V projection
	qkv  *Linear
	proj *Linear

	// Cache: [headIndex][batchIndex] -> (T, headSize)
	kCache [][]*variable.Variable
	vCache [][]*variable.Variable
}

func NewMultiHeadAttention(embedSize, numHeads, blockSize int) *MultiHeadAttention {
	headSize := embedSize / numHeads
	return &MultiHeadAttention{
		numHeads:  numHeads,
		embedSize: embedSize,
		headSize:  headSize,
		blockSize: blockSize,
		// Output size is 3 * embedSize because we want Q, K, V for each head
		qkv:       NewLinear(embedSize, 3*embedSize, NoBias()),
		proj:      NewLinear(embedSize, embedSize, NoBias()),
	}
}

func (mh *MultiHeadAttention) ClearCache() {
	mh.kCache = nil
	mh.vCache = nil
}

func (mh *MultiHeadAttention) Forward(input, cos, sin *variable.Variable, seqLen int, useCache bool) *variable.Variable {
	// input: (B*T, embedSize)
	// qkvOut: (B*T, 3*embedSize)
	qkvOut := mh.qkv.Forward(input)

	// Split into Q, K, V
	qkvT := variable.Transpose(qkvOut)
	splits := pkg.SplitRows(qkvT, 3, mh.embedSize)

	qT, kT, vT := splits[0], splits[1], splits[2]

	// Split columns into heads
	qHeadsT := pkg.SplitRows(qT, mh.numHeads, mh.headSize) // List of (headSize, B*T)
	kHeadsT := pkg.SplitRows(kT, mh.numHeads, mh.headSize)
	vHeadsT := pkg.SplitRows(vT, mh.numHeads, mh.headSize)

	batchSize := len(input.Data) / seqLen
	scale := math.Pow(float64(mh.headSize), -0.5)

	// If useCache is true and cache is nil, initialize it
	if useCache {
		if mh.kCache == nil || len(mh.kCache) != mh.numHeads {
			mh.kCache = make([][]*variable.Variable, mh.numHeads)
			mh.vCache = make([][]*variable.Variable, mh.numHeads)
			for h := 0; h < mh.numHeads; h++ {
				mh.kCache[h] = make([]*variable.Variable, batchSize)
				mh.vCache[h] = make([]*variable.Variable, batchSize)
			}
		}
	}

	qs := make([]*variable.Variable, mh.numHeads)
	ks := make([]*variable.Variable, mh.numHeads)
	vs := make([]*variable.Variable, mh.numHeads)

	for h := 0; h < mh.numHeads; h++ {
		qs[h] = variable.Transpose(qHeadsT[h])
		ks[h] = variable.Transpose(kHeadsT[h])
		vs[h] = variable.Transpose(vHeadsT[h])

		// Apply RoPE
		qs[h] = pkg.ApplyRotaryEmb(qs[h], cos, sin)
		ks[h] = pkg.ApplyRotaryEmb(ks[h], cos, sin)
	}

	// Split by batch
	qsSplit := make([][]*variable.Variable, mh.numHeads)
	ksSplit := make([][]*variable.Variable, mh.numHeads)
	vsSplit := make([][]*variable.Variable, mh.numHeads)

	for h := 0; h < mh.numHeads; h++ {
		qsSplit[h] = pkg.SplitRows(qs[h], batchSize, seqLen)
		ksSplit[h] = pkg.SplitRows(ks[h], batchSize, seqLen)
		vsSplit[h] = pkg.SplitRows(vs[h], batchSize, seqLen)
	}

	var batchOutputs []*variable.Variable

	for b := 0; b < batchSize; b++ {
		var headsInBatch []*variable.Variable
		for h := 0; h < mh.numHeads; h++ {
			q_hb := qsSplit[h][b]
			k_hb := ksSplit[h][b]
			v_hb := vsSplit[h][b]

			if useCache {
				// Detach from graph for caching
				k_detached := variable.NewOf(k_hb.Data...)
				v_detached := variable.NewOf(v_hb.Data...)

				if mh.kCache[h][b] == nil {
					mh.kCache[h][b] = k_detached
					mh.vCache[h][b] = v_detached
				} else {
					mh.kCache[h][b] = pkg.CatV(mh.kCache[h][b], k_detached)
					mh.vCache[h][b] = pkg.CatV(mh.vCache[h][b], v_detached)
				}

				// Crop cache if needed
				if len(mh.kCache[h][b].Data) > mh.blockSize {
					start := len(mh.kCache[h][b].Data) - mh.blockSize
					indices := make([]float64, mh.blockSize)
					for i := 0; i < mh.blockSize; i++ {
						indices[i] = float64(start + i)
					}
					mh.kCache[h][b] = pkg.Rows(mh.kCache[h][b], indices...)
					mh.vCache[h][b] = pkg.Rows(mh.vCache[h][b], indices...)
				}

				// Use cached K, V
				k_hb = mh.kCache[h][b]
				v_hb = mh.vCache[h][b]
			}

			// Attention
			att := MatMul(q_hb, Transpose(k_hb))
			att = MulC(scale, att)

			totalLen := len(k_hb.Data)
			currentLen := len(q_hb.Data)

			if currentLen > 1 {
				// Prompt phase: Causal mask
				tril := Tril(Ones(currentLen, totalLen))
				att = MaskedInfFill(att, tril)
			}

			att = Softmax(att)
			att = Dropout(dropout)(att)

			out_hb := MatMul(att, v_hb)
			headsInBatch = append(headsInBatch, out_hb)
		}
		// Concatenate heads
		batchOut := pkg.Cat(headsInBatch...)
		batchOutputs = append(batchOutputs, batchOut)
	}

	out := pkg.CatV(batchOutputs...)
	out = mh.proj.Forward(out)
	out = Dropout(dropout)(out)

	return out
}

func (mh *MultiHeadAttention) Params() []layer.Parameter {
	return []layer.Parameter{
		mh.qkv.Weight,
		mh.proj.Weight,
	}
}
