package set

import (
	"gonum.org/v1/gonum/graph"
)

// Max is the maximum value of int64.
const Max = int64(^uint64(0) >> 1)

// Ints is a set of int identifiers.
type Ints map[int]struct{}

// The simple accessor methods for Ints are provided to allow ease of
// implementation change should the need arise.

// Add inserts an element into the set.
func (s Ints) Add(e int) {
	s[e] = struct{}{}
}

// Has reports the existence of the element in the set.
func (s Ints) Has(e int) bool {
	_, ok := s[e]
	return ok
}

// Remove deletes the specified element from the set.
func (s Ints) Remove(e int) {
	delete(s, e)
}

// Count reports the number of elements stored in the set.
func (s Ints) Count() int {
	return len(s)
}

// IntsEqual reports set equality between the parameters. Sets are equal if
// and only if they have the same elements.
func IntsEqual(a, b Ints) bool {
	if intsSame(a, b) {
		return true
	}

	if len(a) != len(b) {
		return false
	}

	for e := range a {
		if _, ok := b[e]; !ok {
			return false
		}
	}

	return true
}

// Int64s is a set of int64 identifiers.
type Int64s map[int64]struct{}

// The simple accessor methods for Ints are provided to allow ease of
// implementation change should the need arise.

// Add inserts an element into the set.
func (s Int64s) Add(e int64) {
	s[e] = struct{}{}
}

// Has reports the existence of the element in the set.
func (s Int64s) Has(e int64) bool {
	_, ok := s[e]
	return ok
}

// Remove deletes the specified element from the set.
func (s Int64s) Remove(e int64) {
	delete(s, e)
}

// Count reports the number of elements stored in the set.
func (s Int64s) Count() int {
	return len(s)
}

// Int64sEqual reports set equality between the parameters. Sets are equal if
// and only if they have the same elements.
func Int64sEqual(a, b Int64s) bool {
	if int64sSame(a, b) {
		return true
	}

	if len(a) != len(b) {
		return false
	}

	for e := range a {
		if _, ok := b[e]; !ok {
			return false
		}
	}

	return true
}

// Nodes is a set of nodes keyed in their integer identifiers.
type Nodes map[int64]graph.Node

// The simple accessor methods for Nodes are provided to allow ease of
// implementation change should the need arise.

// Add inserts an element into the set.
func (s Nodes) Add(n graph.Node) {
	s[n.ID()] = n
}

// Remove deletes the specified element from the set.
func (s Nodes) Remove(e graph.Node) {
	delete(s, e.ID())
}

// Has reports the existence of the element in the set.
func (s Nodes) Has(n graph.Node) bool {
	_, ok := s[n.ID()]
	return ok
}

// clear clears the set, possibly using the same backing store.
func (s *Nodes) clear() {
	if len(*s) != 0 {
		*s = make(Nodes)
	}
}

// Copy performs a perfect copy from src to dst (meaning the sets will
// be equal).
func (dst Nodes) Copy(src Nodes) Nodes {
	if same(src, dst) {
		return dst
	}

	if len(dst) > 0 {
		dst = make(Nodes, len(src))
	}

	for e, n := range src {
		dst[e] = n
	}

	return dst
}

// Equal reports set equality between the parameters. Sets are equal if
// and only if they have the same elements.
func Equal(a, b Nodes) bool {
	if same(a, b) {
		return true
	}

	if len(a) != len(b) {
		return false
	}

	for e := range a {
		if _, ok := b[e]; !ok {
			return false
		}
	}

	return true
}

// Union takes the union of a and b, and stores it in dst.
//
// The union of two sets, a and b, is the set containing all the
// elements of each, for instance:
//
//     {a,b,c} UNION {d,e,f} = {a,b,c,d,e,f}
//
// Since sets may not have repetition, unions of two sets that overlap
// do not contain repeat elements, that is:
//
//     {a,b,c} UNION {b,c,d} = {a,b,c,d}
//
func (dst Nodes) Union(a, b Nodes) Nodes {
	if same(a, b) {
		return dst.Copy(a)
	}

	if !same(a, dst) && !same(b, dst) {
		dst.clear()
	}

	if !same(dst, a) {
		for e, n := range a {
			dst[e] = n
		}
	}

	if !same(dst, b) {
		for e, n := range b {
			dst[e] = n
		}
	}

	return dst
}

// Intersect takes the intersection of a and b, and stores it in dst.
//
// The intersection of two sets, a and b, is the set containing all
// the elements shared between the two sets, for instance:
//
//     {a,b,c} INTERSECT {b,c,d} = {b,c}
//
// The intersection between a set and itself is itself, and thus
// effectively a copy operation:
//
//     {a,b,c} INTERSECT {a,b,c} = {a,b,c}
//
// The intersection between two sets that share no elements is the empty
// set:
//
//     {a,b,c} INTERSECT {d,e,f} = {}
//
func (dst Nodes) Intersect(a, b Nodes) Nodes {
	var swap Nodes

	if same(a, b) {
		return dst.Copy(a)
	}
	if same(a, dst) {
		swap = b
	} else if same(b, dst) {
		swap = a
	} else {
		dst.clear()

		if len(a) > len(b) {
			a, b = b, a
		}

		for e, n := range a {
			if _, ok := b[e]; ok {
				dst[e] = n
			}
		}

		return dst
	}

	for e := range dst {
		if _, ok := swap[e]; !ok {
			delete(dst, e)
		}
	}

	return dst
}

// Set implements available ID storage.
type Set struct {
	maxID      int64
	used, free Int64s
}

// NewSet returns a new Set. The returned value should not be passed except by pointer.
func NewSet() Set {
	return Set{maxID: -1, used: make(Int64s), free: make(Int64s)}
}

// NewID returns a new unique ID. The ID returned is not considered used
// until passed in a call to use.
func (s *Set) NewID() int64 {
	for id := range s.free {
		return id
	}
	if s.maxID != Max {
		return s.maxID + 1
	}
	for id := int64(0); id <= s.maxID+1; id++ {
		if !s.used.Has(id) {
			return id
		}
	}
	panic("unreachable")
}

// Use adds the id to the used IDs in the Set.
func (s *Set) Use(id int64) {
	s.used.Add(id)
	s.free.Remove(id)
	if id > s.maxID {
		s.maxID = id
	}
}

// Release frees the id for reuse.
func (s *Set) Release(id int64) {
	s.free.Add(id)
	s.used.Remove(id)
}
