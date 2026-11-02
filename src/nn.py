import torch, torch.nn as nn, pandas as pd, numpy as np
VAN_FACTOR=1.2
class TripNN(nn.Module):
    def __init__(self,n_zone=265):
        super().__init__()
        self.pu=nn.Embedding(n_zone+1,16); self.do=nn.Embedding(n_zone+1,16)
        self.h=nn.Embedding(24,4); self.w=nn.Embedding(7,3)
        self.mlp=nn.Sequential(nn.Linear(16+16+4+3+1,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1))
    def forward(self,pu,do,h,w,d):
        return self.mlp(torch.cat([self.pu(pu),self.do(do),self.h(h),self.w(w),d.unsqueeze(1)],1)).squeeze(1)
def get_tensors(df,mu,sd):
    return (torch.tensor(df.PULocationID.values).long(),torch.tensor(df.DOLocationID.values).long(),
      torch.tensor(df.hour.values).long(),torch.tensor(df.weekday.values).long(),
      torch.tensor(((df.pair_mean_dist-mu)/sd).values).float(),torch.tensor(df.duration.values).float())
def train(epochs=30):
    tr=pd.read_parquet("train_20zones.parquet"); te=pd.read_parquet("test_20zones.parquet")
    mu,sd=float(tr.pair_mean_dist.mean()),float(tr.pair_mean_dist.std()+1e-6)
    trT,teT=get_tensors(tr,mu,sd),get_tensors(te,mu,sd)
    m=TripNN(); opt=torch.optim.Adam(m.parameters(),1e-3,weight_decay=1e-4); fn=nn.L1Loss()
    best=float('inf'); pat=0
    for ep in range(epochs):
        m.train(); perm=torch.randperm(len(trT[0]))
        for i in range(0,len(perm),2048):
            b=perm[i:i+2048]; p=m(*[t[b] for t in trT[:5]]); l=fn(p,trT[5][b]); opt.zero_grad(); l.backward(); opt.step()
        m.eval()
        with torch.no_grad(): v=fn(m(*teT[:5]),teT[5]).item()
        print(ep,round(v,3))
        if v<best: best=v; pat=0; torch.save({"sd":m.state_dict(),"mu":mu,"sd_":sd},"nn.pt")
        else: pat+=1
        if pat>=4: break
    evaluate_baselines(tr,te,teT); export_matrix()
def evaluate_baselines(tr,te,teT):
    # (a) hist-avg per pair-hour (b) distance-based time=dist/18mph*60
    key=list(zip(tr.PULocationID,tr.DOLocationID,tr.hour))
    from collections import defaultdict; import numpy as np
    d=defaultdict(list)
    for k,v in zip(key,tr.duration): d[k].append(v)
    avg={k:np.mean(v) for k,v in d.items()}
    gmean=tr.duration.mean()
    pred_avg=[avg.get((r.PULocationID,r.DOLocationID,r.hour),gmean) for r in te.itertuples()]
    pred_dist=[r.trip_distance/18*60 for r in te.itertuples()]
    ckpt=torch.load("nn.pt"); m=TripNN(); m.load_state_dict(ckpt["sd"]); m.eval()
    with torch.no_grad(): pred_nn=m(*teT[:5]).numpy()
    import pandas as pd
    y=te.duration.values
    def mae(a): return float(np.mean(abs(y-a)))
    def rmse(a): return float(np.sqrt(np.mean((y-a)**2)))
    out=pd.DataFrame([{"model":"NN","MAE":mae(pred_nn),"RMSE":rmse(pred_nn)},{"model":"hist-avg","MAE":mae(pred_avg),"RMSE":rmse(pred_avg)},{"model":"distance","MAE":mae(pred_dist),"RMSE":rmse(pred_dist)}])
    out.to_csv("results/nn_vs_baseline.csv",index=False); print(out)
def export_matrix():
    import json, torch
    ckpt=torch.load("nn.pt"); m=TripNN(); m.load_state_dict(ckpt["sd"]); m.eval(); mu,sd_=ckpt["mu"],ckpt["sd_"]
    top20=json.load(open("top20.json")); pair=pd.read_csv("pair_mean_dist_train.csv")
    dmap={(r.PULocationID,r.DOLocationID):r.pair_mean_dist for r in pair.itertuples()}; gmean=pair.pair_mean_dist.mean()
    import numpy as np; mat=np.zeros((20,20,24))
    with torch.no_grad():
        for i,pu in enumerate(top20):
            for j,do in enumerate(top20):
                dn=(dmap.get((pu,do),gmean)-mu)/sd_
                for h in range(24):
                    mat[i,j,h]=m(torch.tensor([pu]),torch.tensor([do]),torch.tensor([h]),torch.tensor([2]),torch.tensor([dn])).item()*VAN_FACTOR
    np.save("time_matrix_20x20x24.npy",mat)
