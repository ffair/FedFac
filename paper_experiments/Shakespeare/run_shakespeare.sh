cd ../../

# run FedAvg
echo "Run FedAvg"
python mnist_fac2_v2.py --dataset shakespeare --method fed --rounds 100

# run FedProx
echo "Run FedProx"
python mnist_fac2_v2.py --dataset shakespeare --method fedprox --fedprox --mu 0.5 --rounds 100

# run FedPer
echo "Run FedPer"
python mnist_fac2_v2.py --dataset shakespeare --method fedper --last_layer_not_share  --rounds 100

# run LG-FedAvg
echo "Run LG-FedAvg"
python mnist_fac2_v2.py --dataset shakespeare --method lg --partial_cnn_layer 1+2+3+4+5 --rounds 100

# run FedEM
echo "Run FedEM"
python fedEM.py shakespeare FedEM --n_rounds 100 --bz 32 --local_steps 5 --dataset shakespeare 

# run FedFac
echo "Run FedFac"
python mnist_fac2_v2.py --dataset shakespeare --direct_eigDecom --partial_fed --partial_cnn_layer 4 --rounds 100

## Effect of Personalized and Shared parameters
# Personalized parameters
python mnist_fac2_v2.py --dataset shakespeare --direct_eigDecom --partial_fed --partial_cnn_layer 4 --rounds 100 \
--threshold_p --threshold 50 --shared_remove 

# Shared parameters
python mnist_fac2_v2.py --dataset shakespeare --direct_eigDecom --partial_fed --partial_cnn_layer 4 --rounds 100 \
--threshold_p --threshold 50 --private_remove


