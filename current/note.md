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
    next_state <- update_state(next_state, from, to, swap_from, swap_to)
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
