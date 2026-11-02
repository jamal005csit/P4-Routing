# P4-Routing

pip install -r requirements.txt
python make_data.py --month 2023-01 --seed 0
python -m src.nn
python run_demo.py --seed 7
pytest -q
python -m src.eval --exp 1,2,3 --seeds 0,1,2,3,4
