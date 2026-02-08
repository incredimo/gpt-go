package pkg_test

import (
	"testing"

	"github.com/itsubaki/autograd/variable"
	"github.com/zakirullin/gpt-go/pkg"
)

func TestTile(t *testing.T) {
	// (2, 2)
	x := variable.New(
		1, 2,
		3, 4,
	)
	x.Data = [][]float64{{1, 2}, {3, 4}}

	// Tile 3 times -> (6, 2)
	y := pkg.Tile(x, 3)

	if len(y.Data) != 6 {
		t.Errorf("expected 6 rows, got %d", len(y.Data))
	}
	if len(y.Data[0]) != 2 {
		t.Errorf("expected 2 cols, got %d", len(y.Data[0]))
	}

	// Check content
	for i := 0; i < 3; i++ {
		if y.Data[i*2][0] != 1 || y.Data[i*2][1] != 2 {
			t.Errorf("row %d mismatch", i*2)
		}
		if y.Data[i*2+1][0] != 3 || y.Data[i*2+1][1] != 4 {
			t.Errorf("row %d mismatch", i*2+1)
		}
	}
}

func TestSplitRows(t *testing.T) {
	// (4, 2)
	x := variable.New(
		1, 1,
		2, 2,
		3, 3,
		4, 4,
	)
	x.Data = [][]float64{
		{1, 1},
		{2, 2},
		{3, 3},
		{4, 4},
	}

	// Split into 2 chunks of 2 rows
	splits := pkg.SplitRows(x, 2, 2)

	if len(splits) != 2 {
		t.Fatalf("expected 2 splits, got %d", len(splits))
	}

	if len(splits[0].Data) != 2 {
		t.Errorf("expected split 0 to have 2 rows, got %d", len(splits[0].Data))
	}
	if splits[0].Data[0][0] != 1 {
		t.Errorf("expected split 0 row 0 to be {1, 1}, got %v", splits[0].Data[0])
	}
	if splits[1].Data[0][0] != 3 {
		t.Errorf("expected split 1 row 0 to be {3, 3}, got %v", splits[1].Data[0])
	}
}
