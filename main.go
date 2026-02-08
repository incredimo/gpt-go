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
		blocks = append(blocks, NewBlock(cfg.EmbedSize, cfg.Heads, cfg.BlockSize))
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
			embeds = block.Forward(embeds, cosTiled, sinTiled, cfg.BlockSize, false) // useCache=false
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
				e = b.Forward(e, vCos, vSin, cfg.BlockSize, false)
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
	// We use stateful KV cache in blocks.
	nextTok := func(context []float64) float64 {
		// Slice context to max context window
		// Note: With KV cache, we theoretically don't need to re-feed full context if we cropped properly.
		// But usually we just take the new tokens.
		// If context > BlockSize, we crop.
		if len(context) > cfg.BlockSize {
			context = context[len(context)-cfg.BlockSize:]
		}

		// Determine which tokens are new (not yet in cache)
		// For simplicity in this implementation, we assume:
		// 1. If it's the start of generation (prompt), we clear cache and feed full prompt.
		// 2. If it's a subsequent step, we feed only the last token.
		// However, 'nextTok' is called with full growing 'context'.
		// We need to track how much we processed.
		// But this closure is stateless regarding 'processed count'.
		// So we rely on external reset or just logic:
		// Actually, standard practice:
		// The loop calls nextTok with accumulating context.
		// We can change the loop to just pass the new token?
		// But 'data.Encode' returns the prompt.
		// Let's assume 'nextTok' is called correctly.
		// Wait, 'nextTok' inside the loop:
		// context = append(context, nextToken)
		// So context grows.

		// To support KV cache efficiently, we need to know where we are.
		// Let's infer from cache state? No, cache is internal.
		// Let's just process the LAST token if cache is non-empty.
		// BUT if cache is empty, process ALL.

		// We need a way to detect if cache was just cleared.
		// We can check if we are at the beginning of generation.
		// But 'nextTok' is called repeatedly.

		// Let's pass ONLY the tokens we want to process to the model.

		// We need to check if cache is empty. But we can't easily check cache state from here without peeking block internals.
		// Let's add a flag or just assume:
		// If context length == prompt length (first call), process all.
		// If context length > prompt length, process last.
		// BUT we don't know prompt length here easily (it changes).

		// Better approach: Change the loop logic in main.
		// But 'nextTok' encapsulates the model forward pass.

		// Let's simplify:
		// 1. We process full context if we suspect cache is empty.
		//    (We can't really know).
		// 2. We change 'nextTok' to take 'inputTokens' and 'positionOffset'.
		//    And manage the loop explicitly.

		return 0 // Placeholder, logic moved to loop below
	}
	_ = nextTok // silence unused

	// Sample from the model.
	prompt := " mysterious island"
	fmt.Println("Enter prompt (or 'exit'):")
	for {
		fmt.Printf("\n%s", prompt)

		// Reset cache for new prompt
		for _, block := range blocks {
			block.ClearCache()
		}

		context := data.Encode(prompt)

		// First pass: Process full prompt
		inputTokens := context
		startPos := 0

		var nextToken float64

		// Generation loop
		for i := 0; i < cfg.MaxTokens; i++ {
			// Prepare input
			// If inputTokens is empty (should not happen in loop logic), break?

			// RoPE indices
			indices := make([]float64, len(inputTokens))
			for j := 0; j < len(inputTokens); j++ {
				indices[j] = float64(startPos + j)
			}

			// If indices exceed BlockSize, we have a problem with RoPE (precomputed).
			// Our cache logic handles cropping, but RoPE indices must be consistent.
			// Usually RoPE wraps or we clamp.
			// For simplicity, we assume generation doesn't exceed BlockSize from start.
			// If it does, performance degrades.

			cosSlice := Rows(cos, indices...)
			sinSlice := Rows(sin, indices...)

			embeds := Rows(tokEmbeds, inputTokens...)
			for _, block := range blocks {
				embeds = block.Forward(embeds, cosSlice, sinSlice, cfg.BlockSize, true) // useCache=true
			}
			embeds = norm.Forward(embeds)
			logits := lmHead.Forward(embeds) // (T_new, Vocab)

			// Get logits of the LAST token processed
			logitsForNextToken := Rows(logits, -1)

			// Top-K / Top-P Sampling
			// Let's use Top-K = 40, Top-P = 0.9 (common defaults)
			logitsFiltered := pkg.TopK(logitsForNextToken, 40)
			logitsFiltered = pkg.TopP(logitsFiltered, 0.9)

			probs := Softmax(logitsFiltered)
			nextToken = pkg.SampleTemp(probs, 0.8) // Temperature 0.8

			decoded := data.Decode(nextToken)
			fmt.Print(decoded)

			// Prepare for next iteration
			inputTokens = []float64{nextToken}
			startPos += len(indices)

			// Stop if context is too long? Or just let cache crop.
			// Note: startPos increases. If startPos > BlockSize, RoPE will fail (index out of bounds for cos/sin).
			// We must shift RoPE indices or use relative.
			// If we crop cache, we shift the window.
			// But RoPE depends on absolute position.
			// If we crop, we are conceptually sliding the window.
			// The "absolute" position should effectively act as relative to the window start?
			// Standard RoPE uses absolute position.
			// If we exceed precomputed limit, we can't look up cos/sin.
			// We should clamp startPos or extend PrecomputeFreqsCis?
			// Ideally we rotate frequencies for infinite RoPE, but here we just precomputed fixed size.
			if startPos >= cfg.BlockSize {
				// We can't proceed with valid RoPE without recomputing or rotating.
				// For now, let's just stop or wrap (wrapping is bad).
				// We'll break generation.
				break
			}
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
