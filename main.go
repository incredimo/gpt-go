package main

import (
	"bufio"
	"flag"
	"fmt"
	"math"
	"os"
	"strings"

	"github.com/itsubaki/autograd/variable"
	"github.com/zakirullin/gpt-go/data"
	"github.com/zakirullin/gpt-go/pkg"
)

// Hyperparameters
const (
	blockSize        = 32
	embedSize        = 88
	heads            = 4
	layers           = 4
	learningRate     = 0.001 // Higher LR possible due to clipping and scheduler
	steps            = 80000 // number of training steps
	evalSteps        = 1000  // evaluate loss once per every evalSteps
	dropout          = 0.1   // dropout for regularization
	pretrainedTokens = 6000  // number of pretrained tokens
	maxTokens        = 50    // tokens limit for generation
	warmupSteps      = 1000  // steps to warm up learning rate
	gradClip         = 1.0   // maximum gradient norm
	thoughtSteps     = 3     // fixed recurrence: repeat block stack K times for deeper reasoning
	topK             = 40    // top-k sampling: keep only top-K candidates (0 = disabled)
	topP             = 0.9   // top-p (nucleus) sampling: keep smallest set summing to >= p
	repPenalty       = 1.1   // repetition penalty: reduce probability of recent tokens
)

func main() {
	steps := steps
	chat := flag.Bool("chat", false, "Skip training and jump straight to chat")
	flag.Parse()
	if *chat {
		steps = -1
	}

	// Loading dataset and building vocabulary.
	fmt.Println("Tokenizing dataset...")
	dataset, vocabSize := data.Tokenize(pretrainedTokens)
	fmt.Printf("First characters:\n%s\n", strings.TrimSpace(data.Decode(dataset[:45]...)))
	fmt.Printf("Vocabulary: %s\n", data.Chars())
	fmt.Printf("Tokens in dataset: %.3fM\n", pkg.Millions(len(dataset)))

	// --- Model Architecture ---
	tokEmbeds := RandEmbeds(vocabSize, embedSize)
	posEmbeds := RandEmbeds(blockSize, embedSize)
	var blocks []*Block
	for i := range layers {
		blocks = append(blocks, NewBlock(embedSize, heads, i))
	}
	norm := NewLayerNorm(embedSize)

	// Weight Tying: We reuse the token embedding matrix for the final linear layer.
	// This creates a shared latent space for inputs and outputs, significantly
	// reducing the parameter count while improving semantic understanding.
	// We only need a separate bias for the output layer.
	lmHeadBias := Zeros(1, vocabSize)

	// Collecting all the parameters.
	params := pkg.NewParams()
	params.Add(tokEmbeds, posEmbeds)
	for _, block := range blocks {
		params.Add(block.Params()...)
	}
	params.Add(norm.Params()...)
	params.Add(lmHeadBias)
	params.TryLoadPretrained()
	fmt.Printf("Model size: %.3fM\n", pkg.Millions(params.Count()))

	// --- Weight Decay Filtering ---
	// LayerNorm scale/shift, all biases, and embeddings should NOT be
	// weight-decayed. Decaying these hurts training stability.
	noDecay := make(map[*variable.Variable]bool)
	noDecay[tokEmbeds] = true
	noDecay[posEmbeds] = true
	noDecay[lmHeadBias] = true
	noDecay[norm.Scale] = true
	noDecay[norm.Shift] = true
	for _, block := range blocks {
		for _, p := range block.NoDecayParams() {
			noDecay[p] = true
		}
	}

	// Training loop.
	losses := 0.0
	gradNorms := 0.0
	optimizer := pkg.NewAdamW(0)
	optimizer.NoDecay = noDecay

	fmt.Printf("bs=%d, es=%d, vs=%d, steps=%d, thoughts=%d\n", blockSize, embedSize, vocabSize, steps, thoughtSteps)
	for i := 0; i < steps; i++ {
		// --- Cosine Learning Rate Scheduler with Warmup ---
		// Warmup: linear ramp from 0 to learningRate.
		// Decay: cosine curve from learningRate down to 10% of learningRate.
		lr := 0.0
		if i < warmupSteps {
			lr = learningRate * float64(i+1) / float64(warmupSteps)
		} else {
			progress := float64(i-warmupSteps) / float64(steps-warmupSteps)
			lr = learningRate * 0.5 * (1.0 + math.Cos(math.Pi*progress))
		}
		lr = math.Max(lr, learningRate*0.1) // Floor at 10% of peak LR
		optimizer.Alpha = lr

		// Targets contain the ground truth next token for each input token.
		input, targets := data.Sample(dataset, blockSize)

		// --- Forward Pass with Fixed Recurrence ---
		// Instead of one pass through the blocks, we repeat the block stack
		// K times. This provides deeper "thinking" without the instability
		// of adaptive halting. Same weights applied multiple times = parameter efficient.
		embeds := Rows(tokEmbeds, Flat(input)...)
		embeds = Add(embeds, posEmbeds)

		state := embeds
		for t := 0; t < thoughtSteps; t++ {
			for _, block := range blocks {
				state = block.Forward(state)
			}
		}

		state = norm.Forward(state)

		// Weight-tied output projection: state @ tokEmbeds.T + bias
		logits := MatMul(state, Transpose(tokEmbeds))
		logits = Add(logits, lmHeadBias)

		// --- Loss & Backprop ---
		loss := SoftmaxCrossEntropy(logits, targets)
		losses += Val(loss)

		loss.Backward()

		// Gradient Clipping to prevent explosion.
		gradNorm := pkg.ClipGradNorm(params, gradClip)
		gradNorms += gradNorm

		// Nudge parameters to minimize the loss.
		optimizer.Update(params)
		params.ZeroGrad()

		// Logging with grad norm for monitoring training health.
		if i%evalSteps == 0 {
			avgLoss := losses / float64(min(i+1, evalSteps))
			avgGrad := gradNorms / float64(min(i+1, evalSteps))
			fmt.Printf("\rstep: %5d, loss: %.4f, lr: %.6f, grad: %.4f\n", i, avgLoss, lr, avgGrad)
			losses = 0
			gradNorms = 0
		} else if i%100 == 0 {
			fmt.Printf("\r%s", strings.Repeat("·", (i%evalSteps)*26/evalSteps))
		}
	}
	params.Save()
	pkg.DisableDropout()
	// Training is done.

	// --- Inference with Fixed Recurrence ---
	nextTok := func(context []float64) float64 {
		context = context[max(0, len(context)-blockSize):]

		embeds := Rows(tokEmbeds, context...)
		posEmbedsSlice := Rows(posEmbeds, seqIndices(len(context))...)
		embeds = Add(embeds, posEmbedsSlice)

		// Fixed recurrence: think thoughtSteps times.
		state := embeds
		for t := 0; t < thoughtSteps; t++ {
			for _, block := range blocks {
				state = block.Forward(state)
			}
		}

		state = norm.Forward(state)

		// Weight-tied output projection.
		logits := MatMul(state, Transpose(tokEmbeds))
		logits = Add(logits, lmHeadBias)

		// We only care about the next token prediction from the last position.
		logitsForNextToken := Rows(logits, -1)
		probs := Softmax(logitsForNextToken)

		// Advanced sampling: top-k + top-p + repetition penalty.
		tok := pkg.SampleAdvanced(probs, 0.8, topK, topP, context, repPenalty)

		return tok
	}

	// Chat loop.
	prompt := " mysterious island"
	for {
		fmt.Printf("\n%s", prompt)
		context := data.Encode(prompt)
		for i := 0; i < maxTokens; i++ {
			nextToken := nextTok(context)
			fmt.Print(data.Decode(nextToken))
			context = append(context, nextToken)
		}

		fmt.Print("\n$ ")
		scanner := bufio.NewScanner(os.Stdin)
		scanner.Scan()
		prompt = scanner.Text()
		if prompt == "exit" {
			fmt.Println("Bye!")
			break
		}
	}
}

// seqIndices returns a slice of float64 indices [0, 1, 2, ..., n-1].
func seqIndices(n int) []float64 {
	indices := make([]float64, n)
	for i := range indices {
		indices[i] = float64(i)
	}
	return indices
}
