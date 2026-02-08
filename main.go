package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"strings"

	"math"

	"github.com/itsubaki/autograd/optimizer"
	"github.com/zakirullin/gpt-go/data"
	"github.com/zakirullin/gpt-go/pkg"
)

// Hyperparameters
const (
	blockSize        = 64
	embedSize        = 192
	heads            = 6
	layers           = 6
	batchSize        = 32
	learningRate     = 1e-3
	minLearningRate  = 1e-4
	steps            = 10000 // number of training steps
	evalSteps        = 200   // evaluate loss once per every evalSteps
	dropout          = 0.1   // disable some % of our neurons to prevent overfitting, model is likely to generalize
	pretrainedTokens = 6000  // number of pretrained tokens to add on top of auto-detected characters
	maxTokens        = 100   // tokens limit for generation
	maxGradNorm      = 1.0
)

func main() {
	// Skip training if "-chat" flag is provided.
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

	// Basic transformer components.
	tokEmbeds := RandEmbeds(vocabSize, embedSize)
	posEmbeds := RandEmbeds(blockSize, embedSize)
	var blocks []*Block
	for range layers {
		blocks = append(blocks, NewBlock(embedSize, heads))
	}
	norm := NewLayerNorm(embedSize)
	lmHead := NewLinear(embedSize, vocabSize)

	// Collecting all the parameters.
	params := pkg.NewParams()
	params.Add(tokEmbeds, posEmbeds)
	for _, block := range blocks {
		params.Add(block.Params()...)
	}
	params.Add(norm.Params()...)
	params.Add(lmHead.Params()...)
	params.TryLoadPretrained()
	fmt.Printf("Model size: %.3fM\n", pkg.Millions(params.Count()))

	// Training loop.
	losses := 0.0
	opt := pkg.NewAdamW(learningRate)
	fmt.Printf("bs=%d, es=%d, heads=%d, layers=%d, steps=%d\n", blockSize, embedSize, heads, layers, steps)
	for i := 0; i < steps; i++ {
		// Cosine Decay Scheduler
		lr := minLearningRate + 0.5*(learningRate-minLearningRate)*(1+math.Cos(float64(i)/float64(steps)*math.Pi))
		opt.Alpha = lr

		inputs, targets := data.BatchSample(dataset, blockSize, batchSize)
		batchLoss := 0.0

		for j := 0; j < batchSize; j++ {
			input := inputs[j]
			target := targets[j]

			// Forward pass, calculate predictions for every input token.
			embeds := Rows(tokEmbeds, Flat(input)...) // get embed for every input token
			embeds = Add(embeds, posEmbeds)           // add positional embedding
			for _, block := range blocks {            // self-attention and feed-forward
				embeds = block.Forward(embeds)
			}
			embeds = norm.Forward(embeds)
			logits := lmHead.Forward(embeds) // get scores for the next token for every context-enriched embed

			// Loss calculation
			loss := SoftmaxCrossEntropy(logits, target)
			batchLoss += Val(loss)

			// Scale loss by 1/batchSize for gradient accumulation
			scaledLoss := pkg.DivC(float64(batchSize), loss)
			scaledLoss.Backward()
		}

		losses += batchLoss / float64(batchSize)
		fmt.Printf("\r%s", strings.Repeat("·", (i%evalSteps)*26/evalSteps)) // progress bar

		if i%evalSteps == 0 {
			avgLoss := losses / float64(min(i+1, evalSteps))
			fmt.Printf("\rstep: %5d, loss: %.4f, lr: %.5f\n", i, avgLoss, lr)
			losses = 0
		}

		// Gradient Clipping
		paramList := optimizer.Params(params, nil)
		pkg.ClipGradNorm(paramList, maxGradNorm)

		// Nudge the parameters
		opt.Update(params)
		params.ZeroGrad()
	}
	params.Save()
	pkg.DisableDropout()
	// Training is done.

	// Predicts the next token based on the context of tokens.
	nextTok := func(context []float64) float64 {
		context = context[max(0, len(context)-blockSize):]

		// Feed context tokens to the model.
		embeds := Rows(tokEmbeds, context...)
		embeds = Add(embeds, posEmbeds)
		for _, block := range blocks {
			embeds = block.Forward(embeds)
		}
		embeds = norm.Forward(embeds)
		logits := lmHead.Forward(embeds) // get a list of final logits for the next token

		// We only care about the probabilities of the next token for the last token.
		logitsForNextToken := Rows(logits, -1)
		probs := Softmax(logitsForNextToken)
		tok := pkg.SampleTemp(probs, 0.8)

		return tok
	}

	// Sample from the model.
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
