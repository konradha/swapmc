import subprocess


bin_name =  "to_sim"
ts = []
processes = [subprocess.Popen(b, stdout=subprocess.PIPE) for b in 4 * [f"./{bin_name}"]]
for p in processes:
    p.wait()
    print((p.stdout.read()))
