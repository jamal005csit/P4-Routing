import random, json, os

def generate(seed=0, n=20, vans=3, top20=None, depot=162):
    rnd = random.Random(seed)
    if top20 is None:
        if os.path.exists("top20.json"):
            top20 = json.load(open("top20.json"))
        else:
            top20 = [132,138,161,162,163,164,170,186,230,233,234,236,237,239,246,249,261,262,263,141]
    if depot not in top20:
        depot = top20[0]
    orders = []
    for i in range(n):
        z = rnd.choice(top20)
        s = rnd.randint(9*60, 13*60)
        w = rnd.choice([30,45,60])
        orders.append({"id":i,"zone":z,"tw":[s,s+w],"demand":rnd.randint(1,3),"service":3})
    if n > 12:
        z7 = top20[5 % len(top20)]
        z12 = top20[12 % len(top20)]
        if z7 == z12:
            z12 = top20[(12+1) % len(top20)]
        orders[7].update({"tw":[600,630],"zone":z7})
        orders[12].update({"tw":[610,640],"zone":z12})
    # enforce capacity feasible: total <= vans*capacity
    cap = 15
    total = sum(o["demand"] for o in orders)
    while total > vans*cap:
        big = [o for o in orders if o["demand"] > 1]
        if not big: break
        o = rnd.choice(big)
        o["demand"] -= 1
        total -= 1
    inst = {"seed":seed,"n":n,"vans":vans,"depot":depot,
        "depot_start":540,"capacity":cap,"max_stops":10,
        "orders":orders,"top20":top20}
    return inst