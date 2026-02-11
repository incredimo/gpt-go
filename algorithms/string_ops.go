package algorithms

import (
	"fmt"
	"sort"
	"strings"
)

func init() {
	register(&Algorithm{
		ID:          "string.reverse",
		Category:    "string",
		Description: "Reverses a string character by character",
		Complexity:  "O(n)",
		Run:         reverseStringAlgo,
		RandInput:   randomWord,
	})
	register(&Algorithm{
		ID:          "string.palindrome",
		Category:    "string",
		Description: "Checks if a string reads the same forwards and backwards",
		Complexity:  "O(n)",
		Run:         isPalindromeAlgo,
		RandInput: func() string {
			// 50% chance of generating an actual palindrome
			word := randomWord()
			if len(word)%2 == 0 {
				// Make it a palindrome
				runes := []rune(word[:len(word)/2])
				rev := make([]rune, len(runes))
				for i, r := range runes {
					rev[len(runes)-1-i] = r
				}
				return string(runes) + string(rev)
			}
			return word
		},
	})
	register(&Algorithm{
		ID:          "string.charfreq",
		Category:    "string",
		Description: "Counts the frequency of each character in a string",
		Complexity:  "O(n)",
		Run:         charFreqAlgo,
		RandInput:   randomWord,
	})
}

func reverseStringAlgo(args string) (string, []Step, error) {
	s := strings.TrimSpace(args)
	if len(s) == 0 {
		return "", nil, fmt.Errorf("empty input")
	}

	runes := []rune(s)
	var trace []Step

	trace = append(trace, Step{
		Op:     "INIT",
		Detail: fmt.Sprintf("REV(%q)", s),
	})

	left, right := 0, len(runes)-1
	for left < right {
		trace = append(trace, Step{
			Op:     "SWAP",
			Detail: fmt.Sprintf("SWAP([%d]%c,[%d]%c)", left, runes[left], right, runes[right]),
		})
		runes[left], runes[right] = runes[right], runes[left]
		left++
		right--
	}

	result := string(runes)
	trace = append(trace, Step{
		Op:     "DONE",
		Detail: fmt.Sprintf("RESULT=%q", result),
	})

	return result, trace, nil
}

func isPalindromeAlgo(args string) (string, []Step, error) {
	s := strings.TrimSpace(args)
	if len(s) == 0 {
		return "", nil, fmt.Errorf("empty input")
	}

	runes := []rune(strings.ToLower(s))
	var trace []Step

	trace = append(trace, Step{
		Op:     "INIT",
		Detail: fmt.Sprintf("PALINDROME(%q)", s),
	})

	left, right := 0, len(runes)-1
	for left < right {
		if runes[left] != runes[right] {
			trace = append(trace, Step{
				Op:     "MISMATCH",
				Detail: fmt.Sprintf("CMP([%d]%c,[%d]%c)>MISMATCH", left, runes[left], right, runes[right]),
			})
			return "false", trace, nil
		}
		trace = append(trace, Step{
			Op:     "MATCH",
			Detail: fmt.Sprintf("CMP([%d]%c,[%d]%c)>MATCH", left, runes[left], right, runes[right]),
		})
		left++
		right--
	}

	trace = append(trace, Step{Op: "RESULT", Detail: "IS_PALINDROME"})
	return "true", trace, nil
}

func charFreqAlgo(args string) (string, []Step, error) {
	s := strings.TrimSpace(args)
	if len(s) == 0 {
		return "", nil, fmt.Errorf("empty input")
	}

	var trace []Step
	freq := make(map[rune]int)

	for _, ch := range s {
		freq[ch]++
		trace = append(trace, Step{
			Op:     "COUNT",
			Detail: fmt.Sprintf("COUNT(%c)=%d", ch, freq[ch]),
		})
	}

	// Sort keys for deterministic output
	keys := make([]rune, 0, len(freq))
	for k := range freq {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool { return keys[i] < keys[j] })

	var parts []string
	for _, k := range keys {
		parts = append(parts, fmt.Sprintf("%c:%d", k, freq[k]))
	}
	result := strings.Join(parts, ",")

	trace = append(trace, Step{
		Op:     "RESULT",
		Detail: fmt.Sprintf("FREQ={%s}", result),
	})

	return result, trace, nil
}
