package algorithms

import (
	"fmt"
	"strings"
)

// Execute runs an algorithm by ID with the given arguments.
// Returns the result, execution trace, and any error.
//
// Example:
//
//	result, trace, err := Execute("sort.bubble", "5,2,9,1")
//	// result = "1,2,5,9"
//	// trace = [{SWAP CMP(5,2)>SWAP} {KEEP CMP(2,9)>KEEP} ...]
func Execute(algID string, args string) (string, []Step, error) {
	alg, exists := Catalog[algID]
	if !exists {
		return "", nil, fmt.Errorf("unknown algorithm: %q (use ListAlgorithms() to see available)", algID)
	}
	return alg.Run(args)
}

// ParseToolCall checks if a string matches the tool call pattern
// and extracts the algorithm ID and arguments.
//
// Format: <ALGO:algorithm_id:args>
// Example: <ALGO:sort.bubble:5,2,9,1>
func ParseToolCall(s string) (algID string, args string, ok bool) {
	s = strings.TrimSpace(s)
	if !strings.HasPrefix(s, "<ALGO:") || !strings.HasSuffix(s, ">") {
		return "", "", false
	}
	inner := s[6 : len(s)-1] // strip <ALGO: and >
	parts := strings.SplitN(inner, ":", 2)
	if len(parts) != 2 {
		return "", "", false
	}
	return strings.TrimSpace(parts[0]), strings.TrimSpace(parts[1]), true
}

// ExecuteToolCall parses and executes a tool call string.
//
// Example:
//
//	result, trace, err := ExecuteToolCall("<ALGO:sort.bubble:5,2,9,1>")
func ExecuteToolCall(call string) (string, []Step, error) {
	algID, args, ok := ParseToolCall(call)
	if !ok {
		return "", nil, fmt.Errorf("invalid tool call format: %q (expected <ALGO:id:args>)", call)
	}
	return Execute(algID, args)
}

// ListAlgorithms returns a formatted string listing all available algorithms.
func ListAlgorithms() string {
	var sb strings.Builder
	sb.WriteString("Available algorithms:\n")

	// Group by category
	cats := Categories()
	for _, cat := range cats {
		sb.WriteString(fmt.Sprintf("\n  [%s]\n", cat))
		for _, alg := range ByCategory(cat) {
			sb.WriteString(fmt.Sprintf("    %-20s %s (%s)\n", alg.ID, alg.Description, alg.Complexity))
		}
	}

	return sb.String()
}

// FormatTrace formats an execution trace as a compact single-line string.
// This is the "reasoning language" — far more efficient than English.
//
// Example output: "CMP(5,2)>SWAP CMP(2,9)>KEEP CMP(9,1)>SWAP"
func FormatTrace(trace []Step) string {
	parts := make([]string, len(trace))
	for i, s := range trace {
		parts[i] = s.Detail
	}
	return strings.Join(parts, " ")
}

// FormatResult formats an algorithm execution result with its trace
// in a human-readable way.
func FormatResult(algID, args, result string, trace []Step) string {
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("[%s] %s → %s\n", algID, args, result))
	sb.WriteString(fmt.Sprintf("  Trace: %s\n", FormatTrace(trace)))
	return sb.String()
}
