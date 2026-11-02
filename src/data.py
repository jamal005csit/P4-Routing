import random, json
def generate(seed=0,n=20,vans=3,top20=None,depot=162):
    rnd=random.Random(seed)
    top20=top20 or json.load(open("top20.json")) if __import__("os").path.exists("top20.json") else [132,138,161,162,163,164,170,186,230,233,234,236,237,239,246,249,261,262,263,141]
    orders=[]
    for i in range(n):
        z=rnd.choice(top20); s=rnd.randint(9*60,13*60); w=rnd.choice([30,45,60])
        orders.append({"id":i,"zone":z,"tw":[s,s+w],"demand":rnd.randint(1,3),"service":3})
    if n>12: # preserve brief example 7/12 conflict
        orders[7].update({"tw":[600,630],"zone":161}); orders[12].update({"tw":[610,640],"zone":233})
    inst={"seed":seed,"n":n,"vans":vans,"depot":depot,"depot_start":540,"capacity":15,"orders":orders,"top20":top20}
    return inst
