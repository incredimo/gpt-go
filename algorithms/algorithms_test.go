package algorithms

import (
	"strings"
	"testing"
)

func TestBubbleSort(t *testing.T) {
	result, trace, err := Execute("sort.bubble", "5,2,9,1,7")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != "1,2,5,7,9" {
		t.Errorf("want 1,2,5,7,9, got %s", result)
	}
	if len(trace) == 0 {
		t.Error("expected non-empty trace")
	}
}

func TestMergeSort(t *testing.T) {
	result, trace, err := Execute("sort.merge", "8,3,1,5,2")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != "1,2,3,5,8" {
		t.Errorf("want 1,2,3,5,8, got %s", result)
	}
	if len(trace) == 0 {
		t.Error("expected non-empty trace")
	}
}

func TestQuickSort(t *testing.T) {
	result, trace, err := Execute("sort.quick", "4,7,2,1,9,3")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != "1,2,3,4,7,9" {
		t.Errorf("want 1,2,3,4,7,9, got %s", result)
	}
	if len(trace) == 0 {
		t.Error("expected non-empty trace")
	}
}

func TestLinearSearch(t *testing.T) {
	// List: 5,2,9,1  Target: 9 -> found at index 2
	result, trace, err := Execute("search.linear", "5,2,9,1,9")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != "2" {
		t.Errorf("want 2, got %s", result)
	}
	if len(trace) == 0 {
		t.Error("expected non-empty trace")
	}
}

func TestBinarySearch(t *testing.T) {
	// Sorted list: 1,3,5,7,9  Target: 5 -> found
	result, trace, err := Execute("search.binary", "1,3,5,7,9,5")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result == "-1" {
		t.Error("expected to find 5 in sorted list")
	}
	if len(trace) == 0 {
		t.Error("expected non-empty trace")
	}
}

func TestGCD(t *testing.T) {
	result, trace, err := Execute("math.gcd", "48,18")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != "6" {
		t.Errorf("want 6, got %s", result)
	}
	if len(trace) == 0 {
		t.Error("expected non-empty trace")
	}
}

func TestLCM(t *testing.T) {
	result, trace, err := Execute("math.lcm", "12,18")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != "36" {
		t.Errorf("want 36, got %s", result)
	}
	if len(trace) == 0 {
		t.Error("expected non-empty trace")
	}
}

func TestFibonacci(t *testing.T) {
	result, _, err := Execute("math.fibonacci", "10")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != "55" {
		t.Errorf("want 55, got %s", result)
	}
}

func TestFactorial(t *testing.T) {
	result, _, err := Execute("math.factorial", "5")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != "120" {
		t.Errorf("want 120, got %s", result)
	}
}

func TestIsPrime(t *testing.T) {
	tests := map[string]string{
		"2":  "true",
		"7":  "true",
		"10": "false",
		"1":  "false",
		"97": "true",
	}
	for input, want := range tests {
		result, _, err := Execute("math.isprime", input)
		if err != nil {
			t.Fatalf("isPrime(%s): unexpected error: %v", input, err)
		}
		if result != want {
			t.Errorf("isPrime(%s): want %s, got %s", input, want, result)
		}
	}
}

func TestFastPower(t *testing.T) {
	result, _, err := Execute("math.power", "2,10")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != "1024" {
		t.Errorf("want 1024, got %s", result)
	}
}

func TestBFS(t *testing.T) {
	// Simple chain: 0-1-2-3
	result, trace, err := Execute("graph.bfs", "4;0-1,1-2,2-3;0")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.HasPrefix(result, "0,") {
		t.Errorf("BFS should start from node 0, got %s", result)
	}
	if len(trace) == 0 {
		t.Error("expected non-empty trace")
	}
}

func TestDijkstra(t *testing.T) {
	// 0 --4-- 1 --2-- 2
	// |               |
	// +-------7-------+
	// Shortest path 0->2: 0->1->2 = 6 (not direct 7)
	result, trace, err := Execute("graph.dijkstra", "3;0-1:4,1-2:2,0-2:7;0;2")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(result, "dist=6") {
		t.Errorf("want dist=6, got %s", result)
	}
	if len(trace) == 0 {
		t.Error("expected non-empty trace")
	}
}

func TestStringReverse(t *testing.T) {
	result, _, err := Execute("string.reverse", "hello")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != "olleh" {
		t.Errorf("want olleh, got %s", result)
	}
}

func TestIsPalindrome(t *testing.T) {
	tests := map[string]string{
		"racecar": "true",
		"hello":   "false",
		"abba":    "true",
		"a":       "true",
	}
	for input, want := range tests {
		result, _, err := Execute("string.palindrome", input)
		if err != nil {
			t.Fatalf("palindrome(%s): unexpected error: %v", input, err)
		}
		if result != want {
			t.Errorf("palindrome(%s): want %s, got %s", input, want, result)
		}
	}
}

func TestCharFreq(t *testing.T) {
	result, _, err := Execute("string.charfreq", "aabbc")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != "a:2,b:2,c:1" {
		t.Errorf("want a:2,b:2,c:1, got %s", result)
	}
}

func TestToolCallParsing(t *testing.T) {
	algID, args, ok := ParseToolCall("<ALGO:sort.bubble:5,2,9,1>")
	if !ok {
		t.Fatal("expected valid tool call")
	}
	if algID != "sort.bubble" {
		t.Errorf("want sort.bubble, got %s", algID)
	}
	if args != "5,2,9,1" {
		t.Errorf("want 5,2,9,1, got %s", args)
	}
}

func TestToolCallExecution(t *testing.T) {
	result, trace, err := ExecuteToolCall("<ALGO:math.gcd:48,18>")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != "6" {
		t.Errorf("want 6, got %s", result)
	}
	if len(trace) == 0 {
		t.Error("expected non-empty trace")
	}
}

func TestGenerateTraces(t *testing.T) {
	data := GenerateTraces(2)
	if len(data) == 0 {
		t.Error("expected non-empty trace data")
	}
	// Should contain traces for all registered algorithms
	lines := strings.Split(strings.TrimSpace(data), "\n")
	if len(lines) < len(Catalog)*2 {
		t.Errorf("expected at least %d trace lines, got %d", len(Catalog)*2, len(lines))
	}
}

func TestAllAlgorithmsHaveTraces(t *testing.T) {
	for id, alg := range Catalog {
		input := alg.RandInput()
		result, trace, err := alg.Run(input)
		if err != nil {
			t.Errorf("%s with input %q: unexpected error: %v", id, input, err)
			continue
		}
		if result == "" {
			t.Errorf("%s: empty result", id)
		}
		if len(trace) == 0 {
			t.Errorf("%s: empty trace", id)
		}
	}
}

func TestCatalogComplete(t *testing.T) {
	expectedAlgorithms := []string{
		"sort.bubble", "sort.merge", "sort.quick",
		"search.linear", "search.binary",
		"math.gcd", "math.lcm", "math.fibonacci", "math.factorial", "math.isprime", "math.power",
		"graph.bfs", "graph.dijkstra",
		"string.reverse", "string.palindrome", "string.charfreq",
	}

	for _, id := range expectedAlgorithms {
		if _, exists := Catalog[id]; !exists {
			t.Errorf("missing algorithm: %s", id)
		}
	}

	t.Logf("Catalog contains %d algorithms across %d categories",
		len(Catalog), len(Categories()))
}

// --- Q&A Generator Tests ---

func TestGenerateQA(t *testing.T) {
	qa := GenerateQA(10)
	if len(qa) == 0 {
		t.Fatal("expected non-empty Q&A output")
	}

	lines := strings.Split(strings.TrimSpace(qa), "\n")
	// 8 generators * 10 pairs each = 80 lines
	if len(lines) < 80 {
		t.Errorf("expected at least 80 Q&A lines, got %d", len(lines))
	}
}

func TestQAVocabSafe(t *testing.T) {
	// The model's vocabulary is: letters, space, newline, and !',.?
	// Q&A output must ONLY contain these characters.
	allowed := " \n!',.?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
	allowedSet := make(map[rune]bool)
	for _, ch := range allowed {
		allowedSet[ch] = true
	}

	qa := GenerateQA(50) // Large sample for thorough testing

	for i, ch := range qa {
		if !allowedSet[ch] {
			// Find which line the bad character is on
			lineNum := strings.Count(qa[:i], "\n") + 1
			lineStart := strings.LastIndex(qa[:i], "\n") + 1
			lineEnd := strings.Index(qa[i:], "\n")
			if lineEnd == -1 {
				lineEnd = len(qa) - i
			}
			badLine := qa[lineStart : i+lineEnd]
			t.Fatalf("Q&A contains forbidden character '%c' (U+%04X) on line %d: %s",
				ch, ch, lineNum, badLine)
		}
	}

	t.Logf("Q&A output is vocabulary-safe (%d characters verified)", len(qa))
}

func TestQAContainsSortPairs(t *testing.T) {
	qa := GenerateQA(20)
	if !strings.Contains(qa, "Sort ") {
		t.Error("expected Q&A to contain sorting questions")
	}
	if !strings.Contains(qa, "Answer, ") {
		t.Error("expected Q&A to contain answers")
	}
}

func TestQAContainsPalindromeQuestions(t *testing.T) {
	qa := GenerateQA(20)
	if !strings.Contains(qa, "palindrome") {
		t.Error("expected Q&A to contain palindrome questions")
	}
}

func TestQAContainsPrimeQuestions(t *testing.T) {
	qa := GenerateQA(20)
	if !strings.Contains(qa, "prime") {
		t.Error("expected Q&A to contain prime questions")
	}
}

func TestQAContainsReverseQuestions(t *testing.T) {
	qa := GenerateQA(20)
	if !strings.Contains(qa, "Reverse") {
		t.Error("expected Q&A to contain reverse questions")
	}
}

func TestNumberToWordConversion(t *testing.T) {
	tests := map[int]string{
		0: "zero", 1: "one", 5: "five", 10: "ten", 15: "fifteen", 20: "twenty",
	}
	for num, want := range tests {
		got := toWord(num)
		if got != want {
			t.Errorf("toWord(%d): want %s, got %s", num, want, got)
		}
	}
}

func TestWordListFormat(t *testing.T) {
	got := toWordList([]int{3, 1, 7})
	if got != "three, one, seven" {
		t.Errorf("want 'three, one, seven', got '%s'", got)
	}
}
