import subprocess
import threading

bin_name =  "to_sim"
#processes = [subprocess.Popen(f"./{b}") for b in 4 * [bin_name]]
#for p in processes:
#    p.wait()

ts = []
for t in 4 * [f"./{bin_name}"]:
    def proc():
        subprocess.call(t)
    ts.append(threading.Thread(target=proc))
    ts[-1].start()

for t in ts:
    t.join()
