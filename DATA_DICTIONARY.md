# Data Dictionary

This dictionary describes the columns created by `make_data.py` and the generated dispatch instance used by the routing algorithms.

## TLC source columns

| Column | Type / unit | Origin | Meaning |
|---|---|---|---|
| `PULocationID` | integer | Real | TLC pickup taxi-zone identifier. |
| `DOLocationID` | integer | Real | TLC drop-off taxi-zone identifier. |
| `tpep_pickup_datetime` | datetime | Real | Recorded pickup timestamp. |
| `tpep_dropoff_datetime` | datetime | Real | Recorded drop-off timestamp. |
| `trip_distance` | float, miles | Real | Recorded taxi trip distance. |

## Derived dataset columns

| Column | Type / unit | Construction |
|---|---|---|
| `pickup` | datetime | Parsed from `tpep_pickup_datetime`. |
| `dropoff` | datetime | Parsed from `tpep_dropoff_datetime`. |
| `duration` | float, minutes | (`dropoff` - `pickup`) in seconds divided by 60; rows outside 1–120 minutes are removed. |
| `hour` | integer, 0–23 | Hour component of `pickup`. |
| `weekday` | integer, 0–4 | Weekday component of `pickup`; weekends are removed. |
| `day` | integer, 1–31 | Day-of-month component of `pickup`; used for chronological train/test split. |
| `pair_mean_dist` | float, miles | Mean `trip_distance` for each pickup/drop-off pair, computed from training rows only. Test pairs absent from training use the global training mean trip distance. |

## Generated dispatch instance

| Field | Type / unit | Construction |
|---|---|---|
| `orders[].id` | integer | Zero-based order index assigned by the instance generator. |
| `orders[].zone` | integer | Seeded selection from the 20 selected taxi zones. |
| `orders[].tw` | two integers, minutes after midnight | Seeded start between 09:00 and 13:00 and a generated width of 30, 45, or 60 minutes. Seed 7 deliberately sets order 7 to [600, 630] and order 12 to [610, 640] to create a reproducible overlap/conflict demonstration. |
| `orders[].demand` | integer units | Seeded integer demand from 1 to 3. |
| `orders[].service` | minutes | Fixed 3-minute service time. |
| `vans` | integer | Fixed fleet size: 3. |
| `capacity` | integer units per van | Fixed capacity: 15. |
| `max_stops` | integer stops per van | Default limit: 10. |
| `depot` | taxi-zone ID | Fixed depot zone 162, with fallback to the first selected zone if unavailable. |
| `depot_start` | minutes after midnight | Fixed departure time 540 (09:00). |
| `van_factor` | multiplier | Fixed approximation factor 1.2 applied to taxi travel-time estimates when constructing van-scale matrices. |

## Reproducibility and scope

`make_data.py --month 2023-01 --seed 0` downloads the TLC month and uses the supplied seed for reproducible sampling. The split is chronological by day: the first 15 available weekdays are training and later available weekdays are test. Training and test subsets are sampled with `seed` and `seed + 1`, respectively. The generated order instances are separate from the real taxi-trip dataset. Taxi trips are a proxy for van travel, not observed van trips.
