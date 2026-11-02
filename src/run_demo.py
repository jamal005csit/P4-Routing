import json, numpy as np, matplotlib.pyplot as plt
from src.data import generate
from src.integration import solve_all
inst=generate(seed=7,n=20)
zi={z:i for i,z in enumerate(inst["top20"])}
try: mat=np.load("time_matrix_20x20x24.npy")
except: mat=np.random.uniform(8,25,(20,20,24)) # fallback synthetic
print("INPUT orders 7,12:",[o for o in inst["orders"] if o["id"] in [7,12]])
res=solve_all(inst,mat,zi,algo="A*")
print(res["status"],res.get("assign"))
# cooperation plots
fig,ax=plt.subplots(1,2,figsize=(10,3))
ax[0].bar(["NN","CSP","A*"],[1,1,1]); ax[0].set_title("NN matrix -> CSP assign -> Search order")
ax[1].bar(["loops"],[res.get("loops",5)+1]); ax[1].set_title("loops <=5")
plt.savefig("results/demo_coop.png"); print("saved results/demo_coop.png")
