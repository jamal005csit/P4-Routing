# Data Dictionary
## Real (TLC Yellow 2023-01)
- PULocationID int: pickup zone ID. Real.
- DOLocationID int: drop-off zone ID. Real.
- tpep_pickup_datetime datetime: Real.
- tpep_dropoff_datetime datetime: Real.
- trip_distance float miles: Real.
- duration float min = dropoff-pickup, keep 1-120. Derived real.
- hour 0-23 from pickup. Derived real.
- weekday 0-4 Mon-Fri, weekdays only. Derived real.
- pair_mean_dist float: zone-pair mean distance FROM TRAIN ONLY. Derived real, merged to test with fallback global mean.
## Generated (dispatch problem)
- order.zone: choice(top20) seeded rnd. Generated.
- order.tw [start,end]: start uniform 9:00-13:00, width {30,45,60}. Generated. Seed 7 forces 7:[600,630] 12:[610,640] to reproduce 38-min conflict.
- order.demand 1-3 int uniform. Generated.
- order.service 3 min fixed. Generated.
- van capacity 15, vans 3, depot zone 162, depot_start 540 (9:00). Fixed generated.
- van_factor 1.2 multiplies NN time: taxi times only approximate van times. Documented toggle.
