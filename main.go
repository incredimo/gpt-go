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
	maxThoughts      = 8     // maximum number of internal reasoning loops per token
	haltThreshold    = 0.5   // halt probability threshold for inference exit
	ponderCost       = 0.01  // penalty for thinking too long (encourages efficiency)
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

	// The Exit Gate: Determines if the thought is mature enough to be spoken.
	exitGate := NewExitGate(embedSize)

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
	params.Add(exitGate.Params()...)
	params.Add(lmHeadBias) // Only bias; weights are shared with tokEmbeds
	params.TryLoadPretrained()
	fmt.Printf("Model size: %.3fM\n", pkg.Millions(params.Count()))

	// Training loop.
	losses := 0.0
	// Initialize AdamW with 0 learning rate; the scheduler will set it each step.
	optimizer := pkg.NewAdamW(0)
	fmt.Printf("bs=%d, es=%d, vs=%d, steps=%d, thoughts=%d\n", blockSize, embedSize, vocabSize, steps, maxThoughts)

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

		// --- Forward Pass with Latent Reasoning (Pondering Loop) ---
		embeds := Rows(tokEmbeds, Flat(input)...)
		embeds = Add(embeds, posEmbeds)

		// The "Thinking" Process: instead of one pass, loop up to maxThoughts.
		// This allows the model to refine its understanding iteratively in
		// high-dimensional vector space before collapsing to a word.
		state := embeds
		var totalLoss *variable.Variable
		remainProb := variable.New(1.0) // Probability of not having exited yet

		for t := 0; t < maxThoughts; t++ {
			// One step of thinking: pass through all transformer blocks.
			for _, block := range blocks {
				state = block.Forward(state)
			}

			// Introspection: "Am I confident enough to speak?"
			// Exit gate returns (blockSize, 1) halt probabilities.
			haltPerToken := exitGate.Forward(state)

			// Average halt probability across all token positions → scalar.
			haltScalar := Mean(Transpose(haltPerToken))

			// Effective exit probability = P(halt now) * P(haven't exited yet)
			exitProb := Mul(haltScalar, remainProb)

			// Compute what the output WOULD be if we stopped thinking now.
			normState := norm.Forward(state)
			logits := MatMul(normState, Transpose(tokEmbeds)) // Weight tying
			logits = Add(logits, lmHeadBias)
			stepLoss := SoftmaxCrossEntropy(logits, targets)

			// Weight this step's loss by how likely we are to exit here.
			weightedLoss := Mul(stepLoss, exitProb)
			if totalLoss == nil {
				totalLoss = weightedLoss
			} else {
				totalLoss = Add(totalLoss, weightedLoss)
			}

			// Update remaining probability: P(still thinking) *= (1 - P(halt))
			remainProb = Mul(variable.SubC(1.0, haltScalar), remainProb)
		}

		// Ponder cost: penalize the model slightly for using many steps.
		// This encourages efficiency — don't loop forever unnecessarily.
		ponderPenalty := MulC(ponderCost, variable.SubC(1.0, remainProb))
		loss := Add(totalLoss, ponderPenalty)
		losses += Val(loss)

		// --- Backward Pass ---
		loss.Backward()

		// Gradient Clipping to prevent explosion.
		pkg.ClipGradNorm(params, gradClip)

		// Nudge parameters to minimize the loss.
		optimizer.Update(params)
		params.ZeroGrad()

		// Logging
		if i%evalSteps == 0 {
			avgLoss := losses / float64(min(i+1, evalSteps))
			fmt.Printf("\rstep: %5d, loss: %.4f, lr: %.6f\n", i, avgLoss, lr)
			losses = 0
		} else if i%100 == 0 {
			fmt.Printf("\r%s", strings.Repeat("·", (i%evalSteps)*26/evalSteps))
		}
	}
	params.Save()
	pkg.DisableDropout()
	// Training is done.

	// --- Inference with Latent Reasoning ---
	// The model thinks in vector space (loops through blocks) until the
	// exit gate signals confidence, then collapses the thought to a token.
	nextTok := func(context []float64) float64 {
		context = context[max(0, len(context)-blockSize):]

		embeds := Rows(tokEmbeds, context...)
		posEmbedsSlice := Rows(posEmbeds, seqIndices(len(context))...)
		embeds = Add(embeds, posEmbedsSlice)

		// Pondering loop: think until confident or max steps reached.
		state := embeds
		for t := 0; t < maxThoughts; t++ {
			for _, block := range blocks {
				state = block.Forward(state)
			}

			// Check if the model is confident enough to speak.
			haltPerToken := exitGate.Forward(state)
			meanHalt := meanValue(haltPerToken)
			if meanHalt > haltThreshold {
				break // Thought is mature, speak.
			}
		}

		// Collapse thought to token using weight-tied output.
		state = norm.Forward(state)
		logits := MatMul(state, Transpose(tokEmbeds))
		logits = Add(logits, lmHeadBias)

		// We only care about the next token prediction from the last position.
		logitsForNextToken := Rows(logits, -1)
		probs := Softmax(logitsForNextToken)
		tok := pkg.SampleTemp(probs, 0.8)

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

// meanValue computes the arithmetic mean of all elements in a variable.
// Used during inference to get a single scalar halt probability.
func meanValue(x *variable.Variable) float64 {
	sum := 0.0
	count := 0
	for _, row := range x.Data {
		for _, v := range row {
			sum += v
			count++
		}
	}
	if count == 0 {
		return 0
	}
	return sum / float64(count)
}
