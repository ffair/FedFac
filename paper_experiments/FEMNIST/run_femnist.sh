cd ../../

# run FedAvg
echo "Run FedAvg"
python mnist_fac2_v2.py --dataset femnist --num_classes 62 --batch_size 16 --rounds 500 --nettype cnnmnist --local_ep 20 --method fed

# run FedProx
echo "Run FedProx"
python mnist_fac2_v2.py --dataset femnist --num_classes 62 --batch_size 16 --rounds 500 --nettype cnnmnist --method fedprox \
--fedprox --mu 0.5 --local_ep 20

# run FedPer
echo "Run FedPer"
python mnist_fac2_v2.py --dataset femnist --num_classes 62 --batch_size 16 --rounds 500 --nettype cnnmnist --local_ep 20 \
 --method fedper --last_layer_not_share  

# run LG-FedAvg
echo "Run LG-FedAvg"
python mnist_fac2_v2.py --dataset femnist --num_classes 62 --batch_size 16 --rounds 500 --nettype cnnmnist --method lg 
--partial_cnn_layer 1+2 --local_ep 20

# run FedEM
echo "Run FedEM"
python fedEM.py femnist FedEM  --n_learners 3 --n_rounds 500 --bz 16 --local_steps 20  --dataset femnist

# run FedFac
echo "Run FedFac"
python mnist_fac2_v2.py --dataset femnist --num_classes 62 --batch_size 16 --rounds 500 --nettype cnnmnist --partial_fed \
--partial_cnn_layer 2 --direct_eigDecom --local_ep 20 --threshold_p --threshold 90 --cov_thrhd 0.75


## Effect of Personalized and Shared parameters
# Personalized parameters
python mnist_fac2_v2.py --dataset femnist --num_classes 62 --batch_size 16 --rounds 500 --nettype cnnmnist --partial_fed \
--partial_cnn_layer 2 --direct_eigDecom --local_ep 20 --threshold_p --threshold 50 --shared_remove --rounds 500

# Shared parameters
python mnist_fac2_v2.py --dataset femnist --num_classes 62 --batch_size 16 --rounds 500 --nettype cnnmnist --partial_fed \
--partial_cnn_layer 2 --direct_eigDecom --local_ep 20 --threshold_p --threshold 50 --private_remove --rounds 500



