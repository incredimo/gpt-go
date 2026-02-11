// Package algorithms provides a catalog of fundamental algorithmic primitives
// that serve as a "reasoning vocabulary" for the model. Instead of forcing
// the model to simulate algorithms in its weights (expensive, unreliable),
// algorithms are invoked as tools — the model calls them, the system executes.
//
// Each algorithm captures execution traces (step-by-step operations) that can
// be used as training data to teach the model algorithmic reasoning patterns.
//
// Usage:
//
//	result, trace, err := algorithms.Execute("sort.bubble", "5,2,9,1")
//	text := algorithms.GenerateTraces(100) // training data
package algorithms

import (
	"fmt"
	"math/rand/v2"
	"strconv"
	"strings"
)

// Algorithm represents a fundamental algorithmic primitive.
type Algorithm struct {
	ID          string                                                     // Unique identifier, e.g. "sort.bubble"
	Category    string                                                     // Category, e.g. "sorting"
	Description string                                                     // Human-readable description
	Complexity  string                                                     // Time complexity, e.g. "O(n²)"
	Run         func(args string) (result string, trace []Step, err error) // Execute with trace
	RandInput   func() string                                              // Generate a random valid input
}

// Step represents a single operation in an algorithm's execution trace.
// These traces form the "reasoning language" — compact, unambiguous
// descriptions of logical operations that are far more efficient than
// verbose English explanations.
type Step struct {
	Op     string // Operation name: COMPARE, SWAP, VISIT, MERGE, etc.
	Detail string // Compact description: "CMP(5,2)>SWAP"
}

// Catalog holds all registered algorithms, keyed by ID.
var Catalog = make(map[string]*Algorithm)

// Categories returns all unique algorithm categories.
func Categories() []string {
	seen := make(map[string]bool)
	var cats []string
	for _, a := range Catalog {
		if !seen[a.Category] {
			seen[a.Category] = true
			cats = append(cats, a.Category)
		}
	}
	return cats
}

// ByCategory returns all algorithms in a given category.
func ByCategory(category string) []*Algorithm {
	var algs []*Algorithm
	for _, a := range Catalog {
		if a.Category == category {
			algs = append(algs, a)
		}
	}
	return algs
}

// register adds an algorithm to the catalog.
func register(a *Algorithm) {
	Catalog[a.ID] = a
}

// --- Parsing Helpers ---

// parseIntList parses "5,2,9,1" into []int.
func parseIntList(s string) ([]int, error) {
	parts := strings.Split(strings.TrimSpace(s), ",")
	nums := make([]int, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		n, err := strconv.Atoi(p)
		if err != nil {
			return nil, fmt.Errorf("invalid number: %q", p)
		}
		nums = append(nums, n)
	}
	if len(nums) == 0 {
		return nil, fmt.Errorf("empty input")
	}
	return nums, nil
}

// formatIntList formats []int as "1,2,5,9".
func formatIntList(nums []int) string {
	parts := make([]string, len(nums))
	for i, n := range nums {
		parts[i] = strconv.Itoa(n)
	}
	return strings.Join(parts, ",")
}

// randomIntList generates a random comma-separated list of 3-10 integers in [0, 99].
func randomIntList() string {
	n := rand.IntN(8) + 3
	nums := make([]int, n)
	for i := range nums {
		nums[i] = rand.IntN(100)
	}
	return formatIntList(nums)
}

// randomIntPair generates two random integers "a,b" for math operations.
func randomIntPair() string {
	a := rand.IntN(500) + 1
	b := rand.IntN(500) + 1
	return fmt.Sprintf("%d,%d", a, b)
}

// randomSingleInt generates a random integer in [1, 100].
func randomSingleInt() string {
	return strconv.Itoa(rand.IntN(100) + 1)
}

// randomWord generates a random lowercase word of 3-8 characters.
func randomWord() string {
	n := rand.IntN(6) + 3
	chars := make([]byte, n)
	for i := range chars {
		chars[i] = byte('a' + rand.IntN(26))
	}
	return string(chars)
}

// parseTwoArgs splits "arg1,arg2" where arg2 is a single int at the end.
// Used for "list,target" patterns like "5,2,9,1,7" where target=7.
func parseListAndTarget(s string) ([]int, int, error) {
	parts := strings.Split(strings.TrimSpace(s), ",")
	if len(parts) < 2 {
		return nil, 0, fmt.Errorf("need at least list + target")
	}
	target, err := strconv.Atoi(strings.TrimSpace(parts[len(parts)-1]))
	if err != nil {
		return nil, 0, fmt.Errorf("invalid target: %q", parts[len(parts)-1])
	}
	listParts := parts[:len(parts)-1]
	nums := make([]int, len(listParts))
	for i, p := range listParts {
		nums[i], err = strconv.Atoi(strings.TrimSpace(p))
		if err != nil {
			return nil, 0, fmt.Errorf("invalid number: %q", p)
		}
	}
	return nums, target, nil
}
