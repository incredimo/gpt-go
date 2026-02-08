package main

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"strings"

	"github.com/itsubaki/autograd/optimizer"
	"github.com/itsubaki/autograd/variable"
	"github.com/zakirullin/gpt-go/data"
	"github.com/zakirullin/gpt-go/pkg"
)

var dropout float64

func main() {
	cfg := NewConfig()
	dropout = cfg.Dropout

	// Loading dataset and building vocabulary.
	fmt.Println("Tokenizing dataset...")
	fullDataset, vocabSize := data.Tokenize(cfg.PretrainedTokens)
	fmt.Printf("Vocabulary size: %d\n", vocabSize)
	fmt.Printf("Dataset size: %.3fM tokens\n", pkg.Millions(len(fullDataset)))

	// Train/Val split
	trainData, valData := data.SplitDataset(fullDataset, 0.9)
	fmt.Printf("Train: %.3fM, Val: %.3fM\n", pkg.Millions(len(trainData)), pkg.Millions(len(valData)))

	// Basic transformer components.
	tokEmbeds := RandEmbeds(vocabSize, cfg.EmbedSize)
	// RoPE replaces learned posEmbeds

	var blocks []*Block
	for range cfg.Layers {
		blocks = append(blocks, NewBlock(cfg.EmbedSize, cfg.Heads))
	}
	// Use RMSNorm instead of LayerNorm
	norm := NewRMSNorm(cfg.EmbedSize)
	lmHead := NewLinear(cfg.EmbedSize, vocabSize, NoBias())

	// Collecting all the parameters.
	params := pkg.NewParams()
	params.Add(tokEmbeds) // No posEmbeds
	for _, block := range blocks {
		params.Add(block.Params()...)
	}
	params.Add(norm.Params()...)
	params.Add(lmHead.Params()...)
	params.TryLoadPretrained()
	fmt.Printf("Model size: %.3fM params\n", pkg.Millions(params.Count()))

	// Precompute RoPE frequencies
	// Max sequence length is cfg.BlockSize.
	cos, sin := pkg.PrecomputeFreqsCis(cfg.EmbedSize/cfg.Heads, cfg.BlockSize)

	// Training loop.
	losses := 0.0
	opt := pkg.NewAdamW(cfg.LearningRate)

	steps := cfg.Steps
	if cfg.Chat {
		steps = -1
	}

	fmt.Printf("Config: %+v\n", cfg)

	for i := 0; i < steps; i++ {
		// Cosine Decay Scheduler
		lr := cfg.MinLearningRate + 0.5*(cfg.LearningRate-cfg.MinLearningRate)*(1+math.Cos(float64(i)/float64(cfg.Steps)*math.Pi))
		opt.Alpha = lr

		inputs, targets := data.BatchSample(trainData, cfg.BlockSize, cfg.BatchSize)

		// Forward pass
		embeds := Rows(tokEmbeds, Flat(inputs)...) // (B*T, D)

		// Tile RoPE frequencies for the whole batch
		cosTiled := pkg.Tile(cos, cfg.BatchSize)
		sinTiled := pkg.Tile(sin, cfg.BatchSize)

		// Pass RoPE cos/sin to blocks
		for _, block := range blocks {
			embeds = block.Forward(embeds, cosTiled, sinTiled, cfg.BlockSize)
		}
		embeds = norm.Forward(embeds)
		logits := lmHead.Forward(embeds)

		// Loss
		loss := SoftmaxCrossEntropy(logits, targets)
		loss.Backward()

		losses += Val(loss)
		fmt.Printf("\r%s", strings.Repeat("·", (i%cfg.EvalSteps)*26/cfg.EvalSteps))

		if i%cfg.EvalSteps == 0 {
			avgLoss := losses / float64(min(i+1, cfg.EvalSteps))

			// Validation
			pkg.DisableDropout() // Evaluate without dropout
			valLoss := 0.0
			valBatches := 10 // Evaluate on 10 batches

			vInputs, vTargets := data.BatchSample(valData, cfg.BlockSize, valBatches)
			vCos := pkg.Tile(cos, valBatches)
			vSin := pkg.Tile(sin, valBatches)

			e := Rows(tokEmbeds, Flat(vInputs)...)
			for _, b := range blocks {
				e = b.Forward(e, vCos, vSin, cfg.BlockSize)
			}
			e = norm.Forward(e)
			l := lmHead.Forward(e)
			valLoss = Val(SoftmaxCrossEntropy(l, vTargets))

			variable.Config.Train = true // Re-enable dropout

			fmt.Printf("\rstep: %5d, train_loss: %.4f, val_loss: %.4f, lr: %.5f\n", i, avgLoss, valLoss, lr)
			losses = 0

			// Save checkpoint every 5000 steps
			if i > 0 && i%5000 == 0 {
				params.Save()
			}
		}

		paramList := optimizer.Params(params, nil)
		pkg.ClipGradNorm(paramList, cfg.MaxGradNorm)

		opt.Update(params)
		params.ZeroGrad()
	}
	if !cfg.Chat {
		params.Save()
	}
	pkg.DisableDropout()
	fmt.Println("\nTraining completed.")

	// Predicts the next token based on the context of tokens.
	nextTok := func(context []float64) float64 {
		// Slice context to max context window
		if len(context) > cfg.BlockSize {
			context = context[len(context)-cfg.BlockSize:]
		}

		embeds := Rows(tokEmbeds, context...)

		// RoPE slicing
		T := len(context)
		indices := make([]float64, T)
		for i := 0; i < T; i++ {
			indices[i] = float64(i)
		}
		// Assuming cos, sin are large enough. If T < BlockSize (cached size), we can slice.
		// If T > BlockSize, we have a problem. But we capped context to BlockSize above.
		cosSlice := Rows(cos, indices...)
		sinSlice := Rows(sin, indices...)

		for _, block := range blocks {
			embeds = block.Forward(embeds, cosSlice, sinSlice, T)
		}
		embeds = norm.Forward(embeds)
		logits := lmHead.Forward(embeds)

		logitsForNextToken := Rows(logits, -1)
		probs := Softmax(logitsForNextToken)
		tok := pkg.SampleTemp(probs, 0.8)

		return tok
	}

	// Sample from the model.
	prompt := " mysterious island"
	fmt.Println("Enter prompt (or 'exit'):")
	for {
		fmt.Printf("\n%s", prompt)
		context := data.Encode(prompt)
		for i := 0; i < cfg.MaxTokens; i++ {
			nextToken := nextTok(context)
			decoded := data.Decode(nextToken)
			fmt.Print(decoded)
			context = append(context, nextToken)
		}

		fmt.Print("\n$ ")
		scanner := bufio.NewScanner(os.Stdin)
		if !scanner.Scan() {
			break
		}
		prompt = scanner.Text()
		if prompt == "exit" {
			fmt.Println("Bye!")
			break
		}
		if len(strings.TrimSpace(prompt)) == 0 {
			continue
		}
	}
}
