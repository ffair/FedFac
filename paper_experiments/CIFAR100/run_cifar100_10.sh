cd ../../

# run FedAvg
echo "Run FedAvg"
python mnist_fac2_v2.py --dataset cifar100  --sampling pathological_split --n_shards 10 --num_classes 100 \
--weight_decay 0.0005 --method fed --batch_size 64 --local_ep 1 

# run FedProx
echo "Run FedProx"
python mnist_fac2_v2.py --dataset cifar100 --sampling pathological_split --n_shards 10 --num_classes 100 \
--weight_decay 0.0005 --method fedprox --fedprox --mu 0.01 --batch_size 64 --local_ep 1 --rounds 500  

# run FedPer
echo "Run FedPer"
python mnist_fac2_v2.py --dataset cifar100  --sampling pathological_split --n_shards 10 --num_classes 100 \
--weight_decay 0.0005 --method fedper --last_layer_not_share  --batch_size 64 --local_ep 1 --rounds 500

# run LG-FedAvg
echo "Run LG-FedAvg"
python mnist_fac2_v2.py --dataset cifar100  --sampling pathological_split --n_shards 10 --num_classes 100 --weight_decay 0.0005 \
--method lg --partial_cnn_layer 1+2+3+4+5+6+7+8+9+10+11+12+13+14+15  --batch_size 64 --local_ep 1  --rounds 500

# run FedEM
echo "Run FedEM"
python fedEM.py cifar100 FedEM --n_rounds 500 --bz 64 --n_shards 10 --dataset cifar100 

# run FedFac
echo "Run FedFac"
python mnist_fac2_v2.py --dataset cifar100 --sampling pathological_split --n_shards 10 --num_classes 100 \
--weight_decay 0.0005 --direct_eigDecom --partial_fed --partial_cnn_layer 15 --given_threshold --batch_size 64 --local_ep 1 --rounds 500


## Effect of Personalized and Shared parameters
# Personalized parameters
python mnist_fac2_v2.py --dataset cifar100 --sampling pathological_split --n_shards 10 --num_classes 100 --weight_decay 0.0005 \
--direct_eigDecom --partial_fed --partial_cnn_layer 15 --batch_size 64 --local_ep 1 --threshold_p --threshold 50 --shared_remove --rounds 500

# Shared parameters
python mnist_fac2_v2.py --dataset cifar100 --sampling pathological_split --n_shards 10 --num_classes 100 --weight_decay 0.0005 \
--direct_eigDecom --partial_fed --partial_cnn_layer 15 --batch_size 64 --local_ep 1 --threshold_p --threshold 50 --private_remove --rounds 500



