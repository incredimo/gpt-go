package algorithms

import (
	"fmt"
	"math"
	"math/rand/v2"
	"strconv"
	"strings"
)

func init() {
	register(&Algorithm{
		ID:          "math.gcd",
		Category:    "math",
		Description: "Finds the greatest common divisor using Euclid's algorithm",
		Complexity:  "O(log min(a,b))",
		Run:         gcdAlgo,
		RandInput:   randomIntPair,
	})
	register(&Algorithm{
		ID:          "math.lcm",
		Category:    "math",
		Description: "Finds the least common multiple via GCD",
		Complexity:  "O(log min(a,b))",
		Run:         lcmAlgo,
		RandInput:   randomIntPair,
	})
	register(&Algorithm{
		ID:          "math.fibonacci",
		Category:    "math",
		Description: "Computes the N-th Fibonacci number iteratively",
		Complexity:  "O(n)",
		Run:         fibonacciAlgo,
		RandInput: func() string {
			return strconv.Itoa(rand.IntN(20) + 2)
		},
	})
	register(&Algorithm{
		ID:          "math.factorial",
		Category:    "math",
		Description: "Computes N! iteratively",
		Complexity:  "O(n)",
		Run:         factorialAlgo,
		RandInput:   func() string { return strconv.Itoa(rand.IntN(12) + 1) },
	})
	register(&Algorithm{
		ID:          "math.isprime",
		Category:    "math",
		Description: "Tests if a number is prime by trial division",
		Complexity:  "O(√n)",
		Run:         isPrimeAlgo,
		RandInput:   randomSingleInt,
	})
	register(&Algorithm{
		ID:          "math.power",
		Category:    "math",
		Description: "Computes base^exp using fast exponentiation (binary method)",
		Complexity:  "O(log exp)",
		Run:         fastPowerAlgo,
		RandInput: func() string {
			base := rand.IntN(10) + 2
			exp := rand.IntN(10) + 1
			return fmt.Sprintf("%d,%d", base, exp)
		},
	})
}

func gcdAlgo(args string) (string, []Step, error) {
	parts := strings.SplitN(strings.TrimSpace(args), ",", 2)
	if len(parts) != 2 {
		return "", nil, fmt.Errorf("need two numbers: a,b")
	}
	a, err1 := strconv.Atoi(strings.TrimSpace(parts[0]))
	b, err2 := strconv.Atoi(strings.TrimSpace(parts[1]))
	if err1 != nil || err2 != nil {
		return "", nil, fmt.Errorf("invalid numbers")
	}
	if a <= 0 || b <= 0 {
		return "", nil, fmt.Errorf("numbers must be positive")
	}

	var trace []Step
	x, y := a, b
	for y != 0 {
		trace = append(trace, Step{
			Op:     "MOD",
			Detail: fmt.Sprintf("GCD(%d,%d)>MOD=%d", x, y, x%y),
		})
		x, y = y, x%y
	}
	trace = append(trace, Step{
		Op:     "RESULT",
		Detail: fmt.Sprintf("GCD=%d", x),
	})

	return strconv.Itoa(x), trace, nil
}

func lcmAlgo(args string) (string, []Step, error) {
	parts := strings.SplitN(strings.TrimSpace(args), ",", 2)
	if len(parts) != 2 {
		return "", nil, fmt.Errorf("need two numbers: a,b")
	}
	a, err1 := strconv.Atoi(strings.TrimSpace(parts[0]))
	b, err2 := strconv.Atoi(strings.TrimSpace(parts[1]))
	if err1 != nil || err2 != nil {
		return "", nil, fmt.Errorf("invalid numbers")
	}
	if a <= 0 || b <= 0 {
		return "", nil, fmt.Errorf("numbers must be positive")
	}

	var trace []Step

	// First compute GCD
	x, y := a, b
	for y != 0 {
		trace = append(trace, Step{
			Op:     "MOD",
			Detail: fmt.Sprintf("GCD(%d,%d)>MOD=%d", x, y, x%y),
		})
		x, y = y, x%y
	}
	gcd := x
	trace = append(trace, Step{
		Op:     "GCD",
		Detail: fmt.Sprintf("GCD=%d", gcd),
	})

	lcm := (a * b) / gcd
	trace = append(trace, Step{
		Op:     "LCM",
		Detail: fmt.Sprintf("LCM=%d*%d/%d=%d", a, b, gcd, lcm),
	})

	return strconv.Itoa(lcm), trace, nil
}

func fibonacciAlgo(args string) (string, []Step, error) {
	n, err := strconv.Atoi(strings.TrimSpace(args))
	if err != nil {
		return "", nil, fmt.Errorf("invalid number: %q", args)
	}
	if n < 0 {
		return "", nil, fmt.Errorf("n must be non-negative")
	}

	var trace []Step

	if n == 0 {
		trace = append(trace, Step{Op: "BASE", Detail: "FIB(0)=0"})
		return "0", trace, nil
	}
	if n == 1 {
		trace = append(trace, Step{Op: "BASE", Detail: "FIB(1)=1"})
		return "1", trace, nil
	}

	a, b := 0, 1
	trace = append(trace, Step{Op: "INIT", Detail: "F(0)=0,F(1)=1"})
	for i := 2; i <= n; i++ {
		a, b = b, a+b
		trace = append(trace, Step{
			Op:     "STEP",
			Detail: fmt.Sprintf("F(%d)=%d+%d=%d", i, b-a, a, b),
		})
	}

	return strconv.Itoa(b), trace, nil
}

func factorialAlgo(args string) (string, []Step, error) {
	n, err := strconv.Atoi(strings.TrimSpace(args))
	if err != nil {
		return "", nil, fmt.Errorf("invalid number: %q", args)
	}
	if n < 0 {
		return "", nil, fmt.Errorf("n must be non-negative")
	}

	var trace []Step
	result := 1

	if n == 0 {
		trace = append(trace, Step{Op: "BASE", Detail: "0!=1"})
		return "1", trace, nil
	}

	for i := 1; i <= n; i++ {
		result *= i
		trace = append(trace, Step{
			Op:     "MUL",
			Detail: fmt.Sprintf("%d!=%d", i, result),
		})
	}

	return strconv.Itoa(result), trace, nil
}

func isPrimeAlgo(args string) (string, []Step, error) {
	n, err := strconv.Atoi(strings.TrimSpace(args))
	if err != nil {
		return "", nil, fmt.Errorf("invalid number: %q", args)
	}

	var trace []Step

	if n < 2 {
		trace = append(trace, Step{Op: "CHECK", Detail: fmt.Sprintf("%d<2>NOT_PRIME", n)})
		return "false", trace, nil
	}
	if n == 2 {
		trace = append(trace, Step{Op: "CHECK", Detail: "2>PRIME"})
		return "true", trace, nil
	}
	if n%2 == 0 {
		trace = append(trace, Step{Op: "CHECK", Detail: fmt.Sprintf("%d%%2=0>NOT_PRIME", n)})
		return "false", trace, nil
	}

	limit := int(math.Sqrt(float64(n))) + 1
	trace = append(trace, Step{
		Op:     "RANGE",
		Detail: fmt.Sprintf("CHECK_DIVISORS(3..%d)", limit),
	})

	for i := 3; i < limit; i += 2 {
		if n%i == 0 {
			trace = append(trace, Step{
				Op:     "DIVISIBLE",
				Detail: fmt.Sprintf("%d%%%d=0>NOT_PRIME", n, i),
			})
			return "false", trace, nil
		}
		trace = append(trace, Step{
			Op:     "PASS",
			Detail: fmt.Sprintf("%d%%%d!=0>PASS", n, i),
		})
	}

	trace = append(trace, Step{Op: "PRIME", Detail: fmt.Sprintf("%d>PRIME", n)})
	return "true", trace, nil
}

func fastPowerAlgo(args string) (string, []Step, error) {
	parts := strings.SplitN(strings.TrimSpace(args), ",", 2)
	if len(parts) != 2 {
		return "", nil, fmt.Errorf("need two numbers: base,exp")
	}
	base, err1 := strconv.Atoi(strings.TrimSpace(parts[0]))
	exp, err2 := strconv.Atoi(strings.TrimSpace(parts[1]))
	if err1 != nil || err2 != nil {
		return "", nil, fmt.Errorf("invalid numbers")
	}
	if exp < 0 {
		return "", nil, fmt.Errorf("exponent must be non-negative")
	}

	var trace []Step
	result := 1
	b := base
	e := exp

	trace = append(trace, Step{
		Op:     "INIT",
		Detail: fmt.Sprintf("POW(%d,%d)", base, exp),
	})

	for e > 0 {
		if e%2 == 1 {
			result *= b
			trace = append(trace, Step{
				Op:     "MUL",
				Detail: fmt.Sprintf("ODD(%d):result*=%d>%d", e, b, result),
			})
		} else {
			trace = append(trace, Step{
				Op:     "SKIP",
				Detail: fmt.Sprintf("EVEN(%d):skip", e),
			})
		}
		b *= b
		e /= 2
		if e > 0 {
			trace = append(trace, Step{
				Op:     "SQUARE",
				Detail: fmt.Sprintf("BASE²=%d,EXP/2=%d", b, e),
			})
		}
	}

	return strconv.Itoa(result), trace, nil
}
