package algorithms

import (
	"fmt"
	"sort"
	"strings"
)

// GenerateTraces produces training data by running every algorithm in the
// catalog with random inputs and formatting the results as compact text.
//
// Each line is a complete "reasoning trace" — input, step-by-step operations,
// and output. This is the data that teaches a model to "think algorithmically."
//
// Format per line:
//
//	[algo.id] IN: args | STEP1 STEP2 ... | OUT: result
//
// Example:
//
//	[sort.bubble] IN: 5,2,9,1 | CMP(5,2)>SWAP CMP(2,9)>KEEP ... | OUT: 1,2,5,9
func GenerateTraces(tracesPerAlgorithm int) string {
	var sb strings.Builder

	// Sort algorithm IDs for deterministic output order
	ids := make([]string, 0, len(Catalog))
	for id := range Catalog {
		ids = append(ids, id)
	}
	sort.Strings(ids)

	for _, id := range ids {
		alg := Catalog[id]
		for i := 0; i < tracesPerAlgorithm; i++ {
			input := alg.RandInput()
			result, trace, err := alg.Run(input)
			if err != nil {
				continue
			}

			sb.WriteString(fmt.Sprintf("[%s] IN: %s | %s | OUT: %s\n",
				alg.ID, input, FormatTrace(trace), result))
		}
	}

	return sb.String()
}

// GenerateTracesSummary produces a compact summary of the algorithm catalog
// with one example trace per algorithm. Useful for documentation and prompts.
func GenerateTracesSummary() string {
	var sb strings.Builder
	sb.WriteString("=== Algorithm Reasoning Vocabulary ===\n\n")

	ids := make([]string, 0, len(Catalog))
	for id := range Catalog {
		ids = append(ids, id)
	}
	sort.Strings(ids)

	for _, id := range ids {
		alg := Catalog[id]
		input := alg.RandInput()
		result, trace, err := alg.Run(input)
		if err != nil {
			continue
		}

		sb.WriteString(fmt.Sprintf("%-20s %s (%s)\n", alg.ID, alg.Description, alg.Complexity))
		sb.WriteString(fmt.Sprintf("  Example: %s → %s\n", input, result))

		// Show first 5 trace steps
		maxSteps := 5
		if len(trace) < maxSteps {
			maxSteps = len(trace)
		}
		traceStrs := make([]string, maxSteps)
		for i := 0; i < maxSteps; i++ {
			traceStrs[i] = trace[i].Detail
		}
		suffix := ""
		if len(trace) > maxSteps {
			suffix = fmt.Sprintf(" ... (%d more)", len(trace)-maxSteps)
		}
		sb.WriteString(fmt.Sprintf("  Trace:   %s%s\n\n", strings.Join(traceStrs, " → "), suffix))
	}

	return sb.String()
}
