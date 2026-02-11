package algorithms

import (
	"container/heap"
	"fmt"
	"math"
	"math/rand/v2"
	"strconv"
	"strings"
)

func init() {
	register(&Algorithm{
		ID:          "graph.bfs",
		Category:    "graph",
		Description: "Breadth-first search: explores level by level from a source node",
		Complexity:  "O(V+E)",
		Run:         bfsAlgo,
		RandInput:   randomGraphInput,
	})
	register(&Algorithm{
		ID:          "graph.dijkstra",
		Category:    "graph",
		Description: "Finds shortest path in a weighted graph using a priority queue",
		Complexity:  "O((V+E)log V)",
		Run:         dijkstraAlgo,
		RandInput:   randomWeightedGraphInput,
	})
}

// Graph input format for BFS: "numNodes;edge1,edge2,...;startNode"
// Each edge: "a-b" (undirected)
// Example: "5;0-1,1-2,2-3,3-4;0"

// Graph input format for Dijkstra: "numNodes;edge1,edge2,...;source;dest"
// Each edge: "a-b:weight" (undirected, weighted)
// Example: "5;0-1:4,1-2:2,2-3:1;0;3"

type adjList map[int][]struct {
	to, weight int
}

func parseGraph(s string, weighted bool) (int, adjList, int, int, error) {
	parts := strings.Split(strings.TrimSpace(s), ";")
	minParts := 3
	if weighted {
		minParts = 4
	}
	if len(parts) < minParts {
		return 0, nil, 0, 0, fmt.Errorf("invalid graph format")
	}

	numNodes, err := strconv.Atoi(strings.TrimSpace(parts[0]))
	if err != nil {
		return 0, nil, 0, 0, fmt.Errorf("invalid node count: %q", parts[0])
	}

	graph := make(adjList)
	for i := 0; i < numNodes; i++ {
		graph[i] = nil
	}

	edges := strings.Split(strings.TrimSpace(parts[1]), ",")
	for _, e := range edges {
		e = strings.TrimSpace(e)
		if e == "" {
			continue
		}

		var from, to, weight int
		if weighted {
			edgeParts := strings.SplitN(e, ":", 2)
			if len(edgeParts) != 2 {
				return 0, nil, 0, 0, fmt.Errorf("invalid weighted edge: %q", e)
			}
			weight, err = strconv.Atoi(strings.TrimSpace(edgeParts[1]))
			if err != nil {
				return 0, nil, 0, 0, fmt.Errorf("invalid weight: %q", edgeParts[1])
			}
			e = edgeParts[0]
		} else {
			weight = 1
		}

		nodeParts := strings.SplitN(e, "-", 2)
		if len(nodeParts) != 2 {
			return 0, nil, 0, 0, fmt.Errorf("invalid edge: %q", e)
		}
		from, err = strconv.Atoi(strings.TrimSpace(nodeParts[0]))
		if err != nil {
			return 0, nil, 0, 0, fmt.Errorf("invalid node: %q", nodeParts[0])
		}
		to, err = strconv.Atoi(strings.TrimSpace(nodeParts[1]))
		if err != nil {
			return 0, nil, 0, 0, fmt.Errorf("invalid node: %q", nodeParts[1])
		}

		graph[from] = append(graph[from], struct{ to, weight int }{to, weight})
		graph[to] = append(graph[to], struct{ to, weight int }{from, weight})
	}

	source, err := strconv.Atoi(strings.TrimSpace(parts[2]))
	if err != nil {
		return 0, nil, 0, 0, fmt.Errorf("invalid source: %q", parts[2])
	}

	dest := -1
	if len(parts) >= 4 {
		dest, err = strconv.Atoi(strings.TrimSpace(parts[3]))
		if err != nil {
			return 0, nil, 0, 0, fmt.Errorf("invalid dest: %q", parts[3])
		}
	}

	return numNodes, graph, source, dest, nil
}

func randomGraphInput() string {
	n := rand.IntN(4) + 3
	var edges []string
	// Ensure connected by adding chain edges
	for i := 1; i < n; i++ {
		edges = append(edges, fmt.Sprintf("%d-%d", i-1, i))
	}
	// Add some random extra edges
	for i := 0; i < n; i++ {
		for j := i + 2; j < n; j++ {
			if rand.IntN(3) == 0 {
				edges = append(edges, fmt.Sprintf("%d-%d", i, j))
			}
		}
	}
	return fmt.Sprintf("%d;%s;0", n, strings.Join(edges, ","))
}

func randomWeightedGraphInput() string {
	n := rand.IntN(4) + 3
	var edges []string
	for i := 1; i < n; i++ {
		w := rand.IntN(9) + 1
		edges = append(edges, fmt.Sprintf("%d-%d:%d", i-1, i, w))
	}
	for i := 0; i < n; i++ {
		for j := i + 2; j < n; j++ {
			if rand.IntN(3) == 0 {
				w := rand.IntN(9) + 1
				edges = append(edges, fmt.Sprintf("%d-%d:%d", i, j, w))
			}
		}
	}
	return fmt.Sprintf("%d;%s;0;%d", n, strings.Join(edges, ","), n-1)
}

func bfsAlgo(args string) (string, []Step, error) {
	numNodes, graph, source, _, err := parseGraph(args, false)
	if err != nil {
		return "", nil, err
	}

	var trace []Step
	visited := make([]bool, numNodes)
	queue := []int{source}
	visited[source] = true
	var order []int

	trace = append(trace, Step{
		Op:     "START",
		Detail: fmt.Sprintf("BFS(start=%d)", source),
	})

	for len(queue) > 0 {
		node := queue[0]
		queue = queue[1:]
		order = append(order, node)

		trace = append(trace, Step{
			Op:     "VISIT",
			Detail: fmt.Sprintf("VISIT(%d)", node),
		})

		for _, neighbor := range graph[node] {
			if !visited[neighbor.to] {
				visited[neighbor.to] = true
				queue = append(queue, neighbor.to)
				trace = append(trace, Step{
					Op:     "ENQUEUE",
					Detail: fmt.Sprintf("ENQUEUE(%d)", neighbor.to),
				})
			}
		}
	}

	return formatIntList(order), trace, nil
}

// --- Dijkstra's Priority Queue ---

type dijkstraItem struct {
	node int
	dist int
}

type dijkstraPQ []dijkstraItem

func (pq dijkstraPQ) Len() int            { return len(pq) }
func (pq dijkstraPQ) Less(i, j int) bool  { return pq[i].dist < pq[j].dist }
func (pq dijkstraPQ) Swap(i, j int)       { pq[i], pq[j] = pq[j], pq[i] }
func (pq *dijkstraPQ) Push(x interface{}) { *pq = append(*pq, x.(dijkstraItem)) }
func (pq *dijkstraPQ) Pop() interface{} {
	old := *pq
	n := len(old)
	item := old[n-1]
	*pq = old[:n-1]
	return item
}

func dijkstraAlgo(args string) (string, []Step, error) {
	numNodes, graph, source, dest, err := parseGraph(args, true)
	if err != nil {
		return "", nil, err
	}

	var trace []Step

	// Initialize distances to infinity
	dist := make([]int, numNodes)
	prev := make([]int, numNodes)
	for i := range dist {
		dist[i] = math.MaxInt64
		prev[i] = -1
	}
	dist[source] = 0

	pq := &dijkstraPQ{{node: source, dist: 0}}
	heap.Init(pq)

	trace = append(trace, Step{
		Op:     "START",
		Detail: fmt.Sprintf("DIJKSTRA(src=%d,dst=%d)", source, dest),
	})

	for pq.Len() > 0 {
		item := heap.Pop(pq).(dijkstraItem)
		u := item.node

		if item.dist > dist[u] {
			continue // stale entry
		}

		trace = append(trace, Step{
			Op:     "PROCESS",
			Detail: fmt.Sprintf("PROCESS(%d,dist=%d)", u, dist[u]),
		})

		if u == dest {
			trace = append(trace, Step{
				Op:     "REACHED",
				Detail: fmt.Sprintf("DEST_REACHED(dist=%d)", dist[u]),
			})
			break
		}

		for _, edge := range graph[u] {
			v := edge.to
			newDist := dist[u] + edge.weight
			if newDist < dist[v] {
				trace = append(trace, Step{
					Op:     "RELAX",
					Detail: fmt.Sprintf("RELAX(%d→%d:%d<%d)", u, v, newDist, dist[v]),
				})
				dist[v] = newDist
				prev[v] = u
				heap.Push(pq, dijkstraItem{node: v, dist: newDist})
			}
		}
	}

	// Reconstruct path
	if dest >= 0 && dist[dest] < math.MaxInt64 {
		var path []int
		for at := dest; at != -1; at = prev[at] {
			path = append([]int{at}, path...)
		}
		trace = append(trace, Step{
			Op:     "PATH",
			Detail: fmt.Sprintf("PATH(%s)=dist:%d", formatIntList(path), dist[dest]),
		})
		return fmt.Sprintf("dist=%d,path=%s", dist[dest], formatIntList(path)), trace, nil
	}

	// No destination specified, return all distances
	distStrs := make([]string, numNodes)
	for i, d := range dist {
		if d == math.MaxInt64 {
			distStrs[i] = fmt.Sprintf("%d:INF", i)
		} else {
			distStrs[i] = fmt.Sprintf("%d:%d", i, d)
		}
	}
	return strings.Join(distStrs, ","), trace, nil
}
