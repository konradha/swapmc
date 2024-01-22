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


```bash
# pseudocode to save moves per step to disk

print state
i <- 0
while i <= nsteps do
    new_state, from, to = mcstep(lattice)
    x ~ uni(0,1)
    if exp ( -\beta * diff(state, new_state) > x) then
        mc_move(state, from, to)
    end if

    swap_from, swap_to <- -1, -1
    y ~ uni(0,1)
    if y > \theta then                                     # \theta should be tuned
        new_state, from, to = mcswap(lattice)
        x ~ uni(0,1)
        if exp ( -\beta * diff(state, new_state) > x)
            mc_swap(state, from, to)
    end if  
    print i, energy(state), from, to, swap_from, swap_to
    i <- i + 1
    
end while


# pseudocode to calculate autocorrelations for several windows
open file # contains: first line random starting state, next N lines are MC steps
line <- 0
state <- create_state(line)
windows <- {w1, w2, ...}
line <- line + 1
next_state <- create_state(line)
while line != EOF do
    step, energy, from, to, swap_from, swap_to <- line
    next_state <- update_state(next_state, from, to, swap_from, swap_to) # would probably better have a solo-loop
    if line + window >= EOF then
        return
    end if
    
    if step >= window then
        measure(state, new_state)
        state <- update_state(next_state, from, to, swap_from, swap_to) # get right indices here
    end if
        
end while

```


## TODO
- for the second algorithm: devise method to more efficiently walk through the lines
  ie. save all arguments of every line in an array
- also: may have varying length now (EV of \theta -- how many swap moves did we do?)
- second algo can very well be made into a 3d animation


## TODO 2
- check if moves are printed correctly (do better: print arrays at the end to avoid IO)

## TODO 3
- only print out particle moves if != -1 -> lot less data seeing the graphs, no matter the temperature

## TODO 4
- regenerate data: temps .1 .5 1. 1.5 2. 2.5 3. 3.5 and swap .1 with some amount of sweeps
- try to reproduce fig 2a
- burn-in as parameter
- write tests
- introduce sampling with some cutoff around a point

## TODO 5
- write tests to make a convincing case for the parallelization (ie. that it's not faulty)
- check some basic stats (leftmost part of Fig 2a)
- find out how to massively launch jobs on current Euler
- devise strategy to get the measurements we want here...



## statistical tricks to make use of
- sublattice flipping (w/ heat bath method)
- 96 swap mc paper: odd / even numbering of replicas --
- replica exchange
- distributed averaging / majority
- parallel tempering
- particle types constant ?
- cluster updates (swendsen wang?) (how applicable here?)
- more from jap paper


## comments 23.12.
- parallelization now worked out correctly
- when using the "jump-empty"-trick, we get the plateau between 10^1 and 10^3
- when not using that trick, we get inverse behavior of the autocorrelations
- autocorrelation function is still off (by a scaling factor?)
- some parametrized normalization and we would essentially get the same graph as seen in PRL
- TODO: bug fixes, cleanup, read into statistical tricks, how do we approach parallel tempering here?
- read on overlap parameters that we can use to fully exploit massive parallelism we get by using MPI ranks
- TODO: write up on what has been done until now.
- Q: what data to use for RSMI -- logits?
