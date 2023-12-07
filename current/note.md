## PROBLEMS

- saving all configurations is terribly expensive
- writing the entire configuration at timestep t is 
  about an order of magnitude more expensive than the calculations
- OOM when writing to disk the entire thing, need some _smart_
  parallelization etc

## IDEAS
hence, ideas:


1. Reconstructing state 
- save each particle movement for all timesteps t
- reconstruct (to get all autocorrelations we want) the entire
  system's configuration iteartively (depending on window)
- calculate autocorrelation (and other measurements for that matter)
  and dump it in another file


2. producer-consumer architecture to run the simulation:
- introduce a (ring-)buffer that "distributes tasks"
- one thread (affinity 1) does all the calculations and saves moves in an array
- another thread (affinity 5? -- other chip part in all cases)
  writes down moves (we can then reconstruct) to disk
- TODO: benchmark for smaller I/O -- still an order of magnitude?
