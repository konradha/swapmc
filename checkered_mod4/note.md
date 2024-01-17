# "Checkerboard" swap

- checkerboard swap now takes stride = 4
- the inner loop can be collapsed as none of the threads (in any
  dimension) can now overlap
- the strides have to be started 4 * 4 * 4 times 
- this incurs 48 waits / sweep ?
- need some analysis how to make this faster
- maybe sliced sweep + checkerboard can do better
- there's some strong wait penalty caused which is visible in benchmark plots
- advantage: uniform traversal of grid; no bias towards one part/side
- hence: sliced + checkerboard might be optimal
- another thing visible: different temperatures lead to different runtimes /
  advantages when using the non-local swaps

- TODO: compare with sliced sweep
