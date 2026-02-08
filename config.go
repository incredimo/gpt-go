package main

import (
	"flag"
	"fmt"
	"os"
)

type Config struct {
	BlockSize        int
	EmbedSize        int
	Heads            int
	Layers           int
	BatchSize        int
	LearningRate     float64
	MinLearningRate  float64
	Steps            int
	EvalSteps        int
	Dropout          float64
	PretrainedTokens int
	MaxTokens        int
	MaxGradNorm      float64
	Chat             bool
	Resume           string
}

func NewConfig() *Config {
	c := &Config{}

	flag.IntVar(&c.BlockSize, "block-size", 64, "Block size (context length)")
	flag.IntVar(&c.EmbedSize, "embed-size", 192, "Embedding size")
	flag.IntVar(&c.Heads, "heads", 6, "Number of attention heads")
	flag.IntVar(&c.Layers, "layers", 6, "Number of transformer layers")
	flag.IntVar(&c.BatchSize, "batch-size", 32, "Batch size")
	flag.Float64Var(&c.LearningRate, "lr", 1e-3, "Learning rate")
	flag.Float64Var(&c.MinLearningRate, "min-lr", 1e-4, "Minimum learning rate")
	flag.IntVar(&c.Steps, "steps", 10000, "Number of training steps")
	flag.IntVar(&c.EvalSteps, "eval-steps", 200, "Evaluate every N steps")
	flag.Float64Var(&c.Dropout, "dropout", 0.1, "Dropout rate")
	flag.IntVar(&c.PretrainedTokens, "pretrained-tokens", 6000, "Number of pretrained tokens")
	flag.IntVar(&c.MaxTokens, "max-tokens", 100, "Max tokens for generation")
	flag.Float64Var(&c.MaxGradNorm, "max-grad-norm", 1.0, "Max gradient norm")
	flag.BoolVar(&c.Chat, "chat", false, "Skip training and jump straight to chat")
	flag.StringVar(&c.Resume, "resume", "", "Path to checkpoint to resume from")

	flag.Parse()

	if c.EmbedSize%c.Heads != 0 {
		fmt.Printf("Error: embed-size (%d) must be divisible by heads (%d)\n", c.EmbedSize, c.Heads)
		os.Exit(1)
	}

	return c
}
