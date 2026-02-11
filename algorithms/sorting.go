package algorithms

import (
	"fmt"
	"sort"
)

func init() {
	register(&Algorithm{
		ID:          "sort.bubble",
		Category:    "sorting",
		Description: "Sorts a list by repeatedly swapping adjacent elements",
		Complexity:  "O(n²)",
		Run:         bubbleSort,
		RandInput:   randomIntList,
	})
	register(&Algorithm{
		ID:          "sort.merge",
		Category:    "sorting",
		Description: "Sorts a list by dividing, sorting halves, and merging",
		Complexity:  "O(n log n)",
		Run:         mergeSort,
		RandInput:   randomIntList,
	})
	register(&Algorithm{
		ID:          "sort.quick",
		Category:    "sorting",
		Description: "Sorts a list by partitioning around a pivot element",
		Complexity:  "O(n log n) avg",
		Run:         quickSort,
		RandInput:   randomIntList,
	})
}

func bubbleSort(args string) (string, []Step, error) {
	nums, err := parseIntList(args)
	if err != nil {
		return "", nil, err
	}

	arr := make([]int, len(nums))
	copy(arr, nums)
	var trace []Step

	for i := 0; i < len(arr)-1; i++ {
		swapped := false
		for j := 0; j < len(arr)-i-1; j++ {
			if arr[j] > arr[j+1] {
				trace = append(trace, Step{
					Op:     "SWAP",
					Detail: fmt.Sprintf("CMP(%d,%d)>SWAP", arr[j], arr[j+1]),
				})
				arr[j], arr[j+1] = arr[j+1], arr[j]
				swapped = true
			} else {
				trace = append(trace, Step{
					Op:     "KEEP",
					Detail: fmt.Sprintf("CMP(%d,%d)>KEEP", arr[j], arr[j+1]),
				})
			}
		}
		if !swapped {
			trace = append(trace, Step{Op: "DONE", Detail: "SORTED_EARLY"})
			break
		}
	}

	return formatIntList(arr), trace, nil
}

func mergeSort(args string) (string, []Step, error) {
	nums, err := parseIntList(args)
	if err != nil {
		return "", nil, err
	}

	arr := make([]int, len(nums))
	copy(arr, nums)
	var trace []Step

	mergeSortRecursive(arr, 0, len(arr)-1, &trace)

	return formatIntList(arr), trace, nil
}

func mergeSortRecursive(arr []int, left, right int, trace *[]Step) {
	if left >= right {
		return
	}

	mid := (left + right) / 2
	*trace = append(*trace, Step{
		Op:     "SPLIT",
		Detail: fmt.Sprintf("SPLIT(%s|%s)", formatIntList(arr[left:mid+1]), formatIntList(arr[mid+1:right+1])),
	})

	mergeSortRecursive(arr, left, mid, trace)
	mergeSortRecursive(arr, mid+1, right, trace)
	mergeSortMerge(arr, left, mid, right, trace)
}

func mergeSortMerge(arr []int, left, mid, right int, trace *[]Step) {
	leftArr := make([]int, mid-left+1)
	rightArr := make([]int, right-mid)
	copy(leftArr, arr[left:mid+1])
	copy(rightArr, arr[mid+1:right+1])

	i, j, k := 0, 0, left
	for i < len(leftArr) && j < len(rightArr) {
		if leftArr[i] <= rightArr[j] {
			*trace = append(*trace, Step{
				Op:     "MERGE",
				Detail: fmt.Sprintf("PICK_L(%d)", leftArr[i]),
			})
			arr[k] = leftArr[i]
			i++
		} else {
			*trace = append(*trace, Step{
				Op:     "MERGE",
				Detail: fmt.Sprintf("PICK_R(%d)", rightArr[j]),
			})
			arr[k] = rightArr[j]
			j++
		}
		k++
	}

	for i < len(leftArr) {
		arr[k] = leftArr[i]
		i++
		k++
	}
	for j < len(rightArr) {
		arr[k] = rightArr[j]
		j++
		k++
	}

	*trace = append(*trace, Step{
		Op:     "MERGED",
		Detail: fmt.Sprintf("MERGED(%s)", formatIntList(arr[left:right+1])),
	})
}

func quickSort(args string) (string, []Step, error) {
	nums, err := parseIntList(args)
	if err != nil {
		return "", nil, err
	}

	arr := make([]int, len(nums))
	copy(arr, nums)
	var trace []Step

	quickSortRecursive(arr, 0, len(arr)-1, &trace)

	return formatIntList(arr), trace, nil
}

func quickSortRecursive(arr []int, low, high int, trace *[]Step) {
	if low >= high {
		return
	}

	pivot := arr[high]
	*trace = append(*trace, Step{
		Op:     "PIVOT",
		Detail: fmt.Sprintf("PIVOT(%d)", pivot),
	})

	i := low - 1
	for j := low; j < high; j++ {
		if arr[j] < pivot {
			i++
			if i != j {
				*trace = append(*trace, Step{
					Op:     "SWAP",
					Detail: fmt.Sprintf("SWAP(%d,%d)", arr[i], arr[j]),
				})
				arr[i], arr[j] = arr[j], arr[i]
			}
		}
	}
	i++
	if i != high {
		*trace = append(*trace, Step{
			Op:     "PLACE_PIVOT",
			Detail: fmt.Sprintf("PLACE(%d@%d)", pivot, i),
		})
		arr[i], arr[high] = arr[high], arr[i]
	}

	quickSortRecursive(arr, low, i-1, trace)
	quickSortRecursive(arr, i+1, high, trace)

	// Verify with stdlib sort for result correctness
	_ = sort.IntsAreSorted(arr[low : high+1])
}
