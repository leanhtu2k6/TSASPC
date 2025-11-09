python generate_pseudo.py --dataset gtea --feature 1024 --type dp_means
python generate_pseudo.py --dataset gtea  --feature 2048 --type dp_means
python intersection_pseudo.py --dataset gtea --type dp_means
python main.py --action=train --dataset=gtea --split=4