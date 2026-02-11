package algorithms

import (
	"fmt"
	"math/rand/v2"
	"sort"
)

func init() {
	register(&Algorithm{
		ID:          "search.linear",
		Category:    "searching",
		Description: "Finds a target by checking each element sequentially",
		Complexity:  "O(n)",
		Run:         linearSearch,
		RandInput:   randomSearchInput,
	})
	register(&Algorithm{
		ID:          "search.binary",
		Category:    "searching",
		Description: "Finds a target in a sorted list by halving the search space",
		Complexity:  "O(log n)",
		Run:         binarySearch,
		RandInput:   randomSortedSearchInput,
	})
}

// randomSearchInput generates a list with the target as the last element.
// Format: "list...,target" where target is guaranteed to be in the list.
func randomSearchInput() string {
	n := rand.IntN(8) + 3
	nums := make([]int, n)
	for i := range nums {
		nums[i] = rand.IntN(100)
	}
	target := nums[rand.IntN(n)]
	return fmt.Sprintf("%s,%d", formatIntList(nums), target)
}

// randomSortedSearchInput generates a sorted list with target as last element.
func randomSortedSearchInput() string {
	n := rand.IntN(8) + 3
	nums := make([]int, n)
	for i := range nums {
		nums[i] = rand.IntN(100)
	}
	sort.Ints(nums)
	target := nums[rand.IntN(n)]
	return fmt.Sprintf("%s,%d", formatIntList(nums), target)
}

func linearSearch(args string) (string, []Step, error) {
	list, target, err := parseListAndTarget(args)
	if err != nil {
		return "", nil, err
	}

	var trace []Step
	foundIndex := -1

	for i, v := range list {
		if v == target {
			trace = append(trace, Step{
				Op:     "FOUND",
				Detail: fmt.Sprintf("CHECK[%d]=%d>FOUND", i, v),
			})
			foundIndex = i
			break
		}
		trace = append(trace, Step{
			Op:     "SKIP",
			Detail: fmt.Sprintf("CHECK[%d]=%d>SKIP", i, v),
		})
	}

	if foundIndex == -1 {
		trace = append(trace, Step{Op: "NOT_FOUND", Detail: "NOT_FOUND"})
		return "-1", trace, nil
	}

	return fmt.Sprintf("%d", foundIndex), trace, nil
}

func binarySearch(args string) (string, []Step, error) {
	list, target, err := parseListAndTarget(args)
	if err != nil {
		return "", nil, err
	}

	// Binary search requires sorted input
	sort.Ints(list)

	var trace []Step
	low, high := 0, len(list)-1
	foundIndex := -1

	for low <= high {
		mid := (low + high) / 2
		trace = append(trace, Step{
			Op:     "PROBE",
			Detail: fmt.Sprintf("PROBE[%d]=%d(lo=%d,hi=%d)", mid, list[mid], low, high),
		})

		if list[mid] == target {
			trace = append(trace, Step{
				Op:     "FOUND",
				Detail: fmt.Sprintf("FOUND@%d", mid),
			})
			foundIndex = mid
			break
		} else if list[mid] < target {
			trace = append(trace, Step{
				Op:     "GO_RIGHT",
				Detail: fmt.Sprintf("%d<%d>GO_RIGHT", list[mid], target),
			})
			low = mid + 1
		} else {
			trace = append(trace, Step{
				Op:     "GO_LEFT",
				Detail: fmt.Sprintf("%d>%d>GO_LEFT", list[mid], target),
			})
			high = mid - 1
		}
	}

	if foundIndex == -1 {
		trace = append(trace, Step{Op: "NOT_FOUND", Detail: "NOT_FOUND"})
		return "-1", trace, nil
	}

	return fmt.Sprintf("%d", foundIndex), trace, nil
}
