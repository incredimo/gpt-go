<img src="https://raw.githubusercontent.com/MindsMD/minds.md/refs/heads/main/header.svg" alt="gptgo" title="gptgo" align="right" height="60" />

# gpt-go
A modern GPT implementation in pure Go with **latent reasoning** and an **algorithm reasoning vocabulary**. Trained on Jules Verne books.

What kind of response you can expect from the model:  
```
Mysterious Island.
Well.
My days must follow
```

Or this:
```
Captain Nemo, in two hundred thousand feet weary in
the existence of the world.
```

## Architecture

This isn't a vanilla transformer — it implements several breakthroughs from modern LLM research:

| Feature                    | What it does                                                                   |
| -------------------------- | ------------------------------------------------------------------------------ |
| **GELU activation**        | Smoother gradients than ReLU (standard in GPT-2/3, Llama)                      |
| **Weight tying**           | Token embeddings reused as output projection — fewer params, shared semantics  |
| **Cosine LR scheduler**    | Linear warmup → cosine decay for optimal convergence                           |
| **Gradient clipping**      | Prevents exploding gradients, enabling higher learning rates                   |
| **Init scaling**           | Residual projections scaled by 1/√(2·layers) per GPT-2                         |
| **Pre-Norm residual**      | Normalization before attention/MLP, residual on original input                 |
| **AdamW decay filtering**  | LayerNorm, biases, and embeddings excluded from weight decay                   |
| **Fixed recurrence**       | Block stack repeated K times for deeper reasoning (effective 12 layers from 4) |
| **Top-k / Top-p sampling** | Nucleus sampling + repetition penalty for better generation quality            |

## Algorithm Reasoning Vocabulary

The system includes **16 algorithmic primitives** that serve as a "reasoning toolbox." Instead of making the model simulate algorithms in its weights, it can invoke them as tools:

```shell
$ /algo sort.bubble 5,2,9,1
  [sort.bubble] 5,2,9,1 → 1,2,5,9
  Trace: CMP(5,2)>SWAP CMP(2,9)>KEEP CMP(9,1)>SWAP ...

$ /algo graph.dijkstra 3;0-1:4,1-2:2,0-2:7;0;2
  [graph.dijkstra] → dist=6,path=0,1,2

$ /algo math.gcd 48,18
  [math.gcd] 48,18 → 6
  Trace: GCD(48,18)>MOD=12 GCD(18,12)>MOD=6 GCD(12,6)>MOD=0 GCD=6
```

Available algorithms:

| Category      | Algorithms                                          |
| ------------- | --------------------------------------------------- |
| **Sorting**   | Bubble, Merge, Quick                                |
| **Searching** | Linear, Binary                                      |
| **Math**      | GCD, LCM, Fibonacci, Factorial, IsPrime, Fast Power |
| **Graph**     | BFS, Dijkstra (shortest path)                       |
| **String**    | Reverse, Palindrome Check, Character Frequency      |

Each algorithm produces **execution traces** — a compact reasoning language that's 7-10× more efficient than English explanations. These traces can be used as training data:

```shell
$ go run . --traces --trace-count 100
[sort.merge] IN: 82,15,67 | SPLIT(82|15,67) PICK_R(15) PICK_L(82) ... | OUT: 15,67,82
[math.isprime] IN: 97 | CHECK_DIVISORS(3..10) 97%3!=0>PASS 97%5!=0>PASS ... 97>PRIME | OUT: true
```

## How to run
```shell
$ go run .
```

It takes about 40 minutes to train on MacBook Air M3. The trained weights will be saved to `model-1.234M` file. If you rerun the model, it will pick up the saved weights and continue training. The loss should decrease each time, indicating that the model is learning something useful.  

You can train on your own dataset by pointing the `data.dataset` variable to your text corpus.  

To run in chat-only mode once the training is done:  
```shell
$ go run . -chat
```

Chat commands:
```
/list              Browse the full algorithm catalog
/algo <id> <args>  Execute an algorithm (e.g., /algo sort.bubble 5,2,9,1)
/trace <id>        See random execution traces for an algorithm
exit               Quit
```

## How to understand
You can use this repository as a companion to the [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) course. Use `git checkout <tag>` to see how the model has evolved over time: `naive`, `bigram`, `multihead`, `block`, `residual`, `full`.  

In [main_test.go](https://github.com/zakirullin/gpt-go/blob/main/main_test.go) you will find explanations starting from basic neuron example:  
```go
// Our neuron has 2 inputs and 1 output (number of columns in weight matrix).
// Its goal is to predict next number in the sequence.
input := V{1, 2} // {x1, x2}
weight := M{
    {2}, // how much x1 contributes to the output
    {3}, // how much x2 contributes to the output
}
```

All the way to self-attention mechanism:
```go
// To calculate the sum of all previous tokens, we can multiply by this triangular matrix:
tril := M{
    {1, 0, 0, 0}, // first token attends only at itself ("cat"), it can't look into the future
    {1, 1, 0, 0}, // second token attends at itself and the previous token ( "cat" + ", ")
    {1, 1, 1, 0}, // third token attends at itself and the two previous tokens ("cat" + ", " + "dog")
    {1, 1, 1, 1}, // fourth token attends at itself and all the previous tokens ("cat" + ", " + "dog" + " and")
}.Var()
// So, at this point each embedding is enriched with the information from all the previous tokens.
// That's the crux of self-attention.
enrichedEmbeds := MatMul(tril, inputEmbeds)
```

## Design choices
No batches.  
I've given up the complexity of the batch dimension for the sake of better understanding. It's far easier to build intuition with 2D matrices, rather than with 3D tensors. Besides, batches aren't inherent to the transformer architecture. For better gradient smoothing gradient accumulation was tried. The effect was negligible, so it was removed as well.   

Removed `gonum`.  
The `gonum.matmul` gave us ~30% performance boost, but it brought additional dependency. We're not striving for maximum efficiency here, rather for radical simplicity. Current matmul implementation is quite effective, and it's only 40 lines of plain readable code.  

## Papers
You don't need to read them to understand the code :)  

[Attention Is All You Need](https://arxiv.org/abs/1706.03762)  
[Deep Residual Learning](https://arxiv.org/abs/1512.03385)  
[DeepMind WaveNet](https://arxiv.org/abs/1609.03499)  
[Batch Normalization](https://arxiv.org/abs/1502.03167)  
[Deep NN + huge data = breakthrough performance](https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)  
[OpenAI GPT-3 paper](https://arxiv.org/abs/2005.14165)  
[Analyzing the Structure of Attention](https://arxiv.org/abs/1906.04284)  
[Gaussian Error Linear Units (GELUs)](https://arxiv.org/abs/1606.08415)  
[Universal Transformers](https://arxiv.org/abs/1807.03819)  
[Adaptive Computation Time](https://arxiv.org/abs/1603.08983)  

## Credits
Many thanks to [Andrej Karpathy](https://github.com/karpathy) for his brilliant [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) course.

Thanks to [@itsubaki](https://github.com/itsubaki) for his elegant [autograd](https://github.com/itsubaki/autograd) package.
