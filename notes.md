2023-11-23
- we're mostly interested in temporal autocorrelation:
- interesting would be to find a power law (indicating
  glassy dynamics)
- C(r) \prop exp(-r/alpha) versus C(r) \prop r^{-alpha}
- maybe parallelization over some local radius would be nice (stencil):
  take local (spatial!) autocorrelates and evaluate
- parallel tempering: run several in parallel: but change some parameter
  -> ie. temperature (what's more interesting -- maybe -- for us, is to 
     vary the density of the system in parallel tempering)
- in japanese paper: autocorrelates mimics "spin autocorrelation" (kronecker delta)


## TODOs
- [ ] finish autocorrelation investigation
- [ ] reproduce glassy phase relaxation from japanese paper 
- [ ] submit jobs to do the grid search: density / temperature
- [ ] more profiling to get most possible optimizations done
- [ ] develop strategy to save configurations (do we need all of them?)
- [ ] parallel tempering tries
- [ ] susceptibility graphs (need _lots_ of data) (as higher moments are harder to estimate)
- [ ] write down first part of investigations: what is a glass, why are glassy dynamics interesting
      why are they hard, can we say something about the unreasonable effectiveness of swap mc,
      'modern' code -- ie. C implementation with best practices from HPC still the fastest?, 
      power laws versus superexponentials, useful theory: information bottleneck method, measure theory,
      KL divergence and other useful results from information theory,  
 
- [ ] papers to investigate (relate?): RG-Flow (explaining RG ...), Bouchaud recent, glass transition
      experimental, relevant readings (+exercises) from `Information, Physics and Computation`, Montanari:
      `glass-with-exercise.pdf`, more theory: "Thermodynamic signature of growing amorphous order in
      glass-forming liquids". rigorous inequalities, szilard entropieminderung, landauer: irreversibility,
      How many clusters?, `Information Bottleneck Approach to Predictive Inference`, overlap functions,
      fluctuation theorems (both in parisi's book on spin glasses...)
