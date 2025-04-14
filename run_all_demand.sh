# sg hourly
python run_demand.py --target_dataset demand_sg --dataset_freq 1h
python run_demand.py --target_dataset demand_sg_feat --dataset_freq 1h
python run_demand.py --target_dataset demand_sg_top0 --dataset_freq 1h

# sg daily
python run_demand.py --target_dataset demand_sg_daily --dataset_freq D
python run_demand.py --target_dataset demand_sg_daily_feat --dataset_freq D
python run_demand.py --target_dataset demand_sg_daily_top0 --dataset_freq D

# aus hourly
python run_demand.py --target_dataset demand_aus --dataset_freq 1h
python run_demand.py --target_dataset demand_aus_feat --dataset_freq 1h
python run_demand.py --target_dataset demand_aus_top0 --dataset_freq 1h

# aus daily
python run_demand.py --target_dataset demand_aus_daily --dataset_freq D
python run_demand.py --target_dataset demand_aus_daily_feat --dataset_freq D
python run_demand.py --target_dataset demand_aus_daily_top0 --dataset_freq D
