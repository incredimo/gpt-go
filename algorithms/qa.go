package algorithms

import (
	"fmt"
	"math/rand/v2"
	"sort"
	"strings"
)

// Number words for generating vocabulary-safe training data.
// The model's vocabulary is character-level from literary text, so it knows
// letters but not digits. We spell numbers as words so the model can learn.
var numWord = []string{
	"zero", "one", "two", "three", "four", "five",
	"six", "seven", "eight", "nine", "ten",
	"eleven", "twelve", "thirteen", "fourteen", "fifteen",
	"sixteen", "seventeen", "eighteen", "nineteen", "twenty",
}

// wordToNum maps word back to number for verification.
var wordToNum map[string]int

func init() {
	wordToNum = make(map[string]int, len(numWord))
	for i, w := range numWord {
		wordToNum[w] = i
	}
}

// toWord converts a small integer to its word form.
func toWord(n int) string {
	if n >= 0 && n < len(numWord) {
		return numWord[n]
	}
	return fmt.Sprintf("%d", n) // fallback for numbers outside range
}

// toWordList converts a slice of ints to comma-separated words.
func toWordList(nums []int) string {
	words := make([]string, len(nums))
	for i, n := range nums {
		words[i] = toWord(n)
	}
	return strings.Join(words, ", ")
}

// GenerateQA produces natural language question-answer pairs that teach
// the model to RECOGNIZE algorithmic problems and their solutions.
//
// Unlike raw execution traces (full of digits and brackets that the model
// can't tokenize), these Q&A pairs use only characters already in the
// vocabulary. The model learns semantic associations, not execution.
//
// Example output:
//
//	Sort three, seven, one, nine. Answer, one, three, seven, nine.
//	Is racecar a palindrome? Yes.
//	Reverse the word apple. Answer, elppa.
func GenerateQA(pairsPerType int) string {
	var sb strings.Builder

	generators := []func(*strings.Builder){
		generateSortQA,
		generateReverseQA,
		generatePalindromeQA,
		generatePrimeQA,
		generateFactorialQA,
		generateFibonacciQA,
		generateGCDQA,
		generateCompareQA,
	}

	for _, gen := range generators {
		for i := 0; i < pairsPerType; i++ {
			gen(&sb)
		}
	}

	return sb.String()
}

func generateSortQA(sb *strings.Builder) {
	n := rand.IntN(4) + 3 // 3-6 numbers
	nums := make([]int, n)
	for i := range nums {
		nums[i] = rand.IntN(20) + 1 // 1-20
	}

	sorted := make([]int, n)
	copy(sorted, nums)
	sort.Ints(sorted)

	sb.WriteString(fmt.Sprintf("Sort %s. Answer, %s.\n",
		toWordList(nums), toWordList(sorted)))
}

func generateReverseQA(sb *strings.Builder) {
	words := []string{
		"apple", "hello", "world", "ocean", "river",
		"storm", "dream", "light", "stone", "flame",
		"island", "captain", "voyage", "forest", "castle",
		"mountain", "thunder", "crystal", "shadow", "garden",
	}
	word := words[rand.IntN(len(words))]

	runes := []rune(word)
	for i, j := 0, len(runes)-1; i < j; i, j = i+1, j-1 {
		runes[i], runes[j] = runes[j], runes[i]
	}
	reversed := string(runes)

	sb.WriteString(fmt.Sprintf("Reverse the word %s. Answer, %s.\n", word, reversed))
}

func generatePalindromeQA(sb *strings.Builder) {
	palindromes := []string{
		"racecar", "level", "madam", "rotor", "civic",
		"kayak", "refer", "noon", "deed", "tenet",
	}
	nonPalindromes := []string{
		"hello", "world", "ocean", "river", "storm",
		"dream", "light", "stone", "flame", "apple",
	}

	if rand.IntN(2) == 0 {
		word := palindromes[rand.IntN(len(palindromes))]
		sb.WriteString(fmt.Sprintf("Is %s a palindrome? Yes.\n", word))
	} else {
		word := nonPalindromes[rand.IntN(len(nonPalindromes))]
		sb.WriteString(fmt.Sprintf("Is %s a palindrome? No.\n", word))
	}
}

func generatePrimeQA(sb *strings.Builder) {
	primes := []int{2, 3, 5, 7, 11, 13, 17, 19}
	nonPrimes := []int{4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20}

	if rand.IntN(2) == 0 {
		p := primes[rand.IntN(len(primes))]
		sb.WriteString(fmt.Sprintf("Is %s prime? Yes.\n", toWord(p)))
	} else {
		np := nonPrimes[rand.IntN(len(nonPrimes))]
		sb.WriteString(fmt.Sprintf("Is %s prime? No.\n", toWord(np)))
	}
}

func generateFactorialQA(sb *strings.Builder) {
	// Only use small factorials that have reasonable word representations.
	factorials := map[int]string{
		1: "one",
		2: "two",
		3: "six",
		4: "twenty four",
		5: "one hundred twenty",
		6: "seven hundred twenty",
	}

	keys := []int{1, 2, 3, 4, 5, 6}
	k := keys[rand.IntN(len(keys))]
	sb.WriteString(fmt.Sprintf("What is %s factorial? Answer, %s.\n", toWord(k), factorials[k]))
}

func generateFibonacciQA(sb *strings.Builder) {
	// Fibonacci numbers small enough to spell.
	fibs := map[int]string{
		1: "one", 2: "one", 3: "two", 4: "three", 5: "five",
		6: "eight", 7: "thirteen", 8: "twenty one",
	}

	keys := []int{1, 2, 3, 4, 5, 6, 7, 8}
	k := keys[rand.IntN(len(keys))]
	ordinal := []string{"", "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth"}
	sb.WriteString(fmt.Sprintf("What is the %s Fibonacci number? Answer, %s.\n", ordinal[k], fibs[k]))
}

func generateGCDQA(sb *strings.Builder) {
	// Pairs with known GCDs that spell nicely.
	type gcdPair struct {
		a, b, gcd int
	}
	pairs := []gcdPair{
		{12, 8, 4}, {15, 10, 5}, {18, 12, 6}, {20, 15, 5},
		{9, 6, 3}, {14, 7, 7}, {16, 12, 4}, {10, 4, 2},
		{6, 4, 2}, {8, 6, 2}, {15, 5, 5}, {12, 4, 4},
	}
	p := pairs[rand.IntN(len(pairs))]
	sb.WriteString(fmt.Sprintf("What is the greatest common divisor of %s and %s? Answer, %s.\n",
		toWord(p.a), toWord(p.b), toWord(p.gcd)))
}

func generateCompareQA(sb *strings.Builder) {
	a := rand.IntN(19) + 1 // 1-19
	b := a
	for b == a {
		b = rand.IntN(19) + 1
	}

	if a > b {
		sb.WriteString(fmt.Sprintf("Which is larger, %s or %s? Answer, %s.\n",
			toWord(a), toWord(b), toWord(a)))
	} else {
		sb.WriteString(fmt.Sprintf("Which is larger, %s or %s? Answer, %s.\n",
			toWord(a), toWord(b), toWord(b)))
	}
}
