import argparse, hashlib, pandas as pd
PARQUET="https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet"
LOOKUP="https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
COLS=["PULocationID","DOLocationID","tpep_pickup_datetime","tpep_dropoff_datetime","trip_distance"]
def run(month="2023-01", seed=0):
    url=f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{month}.parquet"
    df=pd.read_parquet(url, columns=COLS)
    assert all(c in df.columns for c in COLS), f"missing zone-ID columns, got {list(df.columns)}"
    df["pickup"]=pd.to_datetime(df["tpep_pickup_datetime"])
    df["dropoff"]=pd.to_datetime(df["tpep_dropoff_datetime"])
    df["duration"]=(df["dropoff"]-df["pickup"]).dt.total_seconds()/60
    df=df[(df.duration>=1)&(df.duration<=120)]
    df["hour"]=df.pickup.dt.hour
    df["weekday"]=df.pickup.dt.weekday
    df["day"]=df.pickup.dt.day
    df=df[df.weekday<5]
    top20=df["PULocationID"].value_counts().head(20).index.tolist()
    df=df[df.PULocationID.isin(top20)&df.DOLocationID.isin(top20)]
    days=sorted(df.day.unique())
    train_days=set(days[:15]) # 3 weeks train, last week test
    train=df[df.day.isin(train_days)].copy()
    test=df[~df.day.isin(train_days)].copy()
    pair=train.groupby(["PULocationID","DOLocationID"])["trip_distance"].mean().reset_index().rename(columns={"trip_distance":"pair_mean_dist"})
    train=train.merge(pair,on=["PULocationID","DOLocationID"],how="left")
    test=test.merge(pair,on=["PULocationID","DOLocationID"],how="left")
    test["pair_mean_dist"]=test["pair_mean_dist"].fillna(train["trip_distance"].mean())
    # reproducible subset 300k / 30k
    train=train.sample(n=min(300000,len(train)),random_state=seed)
    test=test.sample(n=min(30000,len(test)),random_state=seed+1)
    train.to_parquet("train_20zones.parquet"); test.to_parquet("test_20zones.parquet")
    pair.to_csv("pair_mean_dist_train.csv",index=False)
    import json; json.dump(top20,open("top20.json","w"))
    print(f"TRAIN {len(train)} TEST {len(test)} TOP20 {top20}")
    print("sha train",hashlib.md5(str(len(train)).encode()).hexdigest())
if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--month",default="2023-01"); ap.add_argument("--seed",type=int,default=0)
    a=ap.parse_args(); run(a.month,a.seed)
