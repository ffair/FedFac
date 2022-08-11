cd ../../

# run FedAvg
echo "Run FedAvg"
python mnist_fac2_v2.py --dirichlet_parameter 1 --rounds 500 --bz 32 --nettype pre_resnet18 --weight_decay 0.00001 \
--method fed --local_ep 20 

# run FedProx
echo "Run FedProx"
python mnist_fac2_v2.py --dirichlet_parameter 1 --rounds 500 --bz 32 --nettype pre_resnet18 --weight_decay 0.00001 \
--method fedprox --fedprox --mu 0.001 --local_ep 20

# run FedPer
echo "Run FedPer"
python mnist_fac2_v2.py --rounds 500 --bz 32 --nettype pre_resnet18 --weight_decay 0.00001 --method fedper --last_layer_not_share \
--local_ep 20 --dirichlet_parameter 1 

# run LG-FedAvg
echo "Run LG-FedAvg"
python mnist_fac2_v2.py --dirichlet_parameter 1 --rounds 500 --bz 32 --nettype pre_resnet18 --weight_decay 0.00001 --method lg \
--partial_cnn_layer 1+2+3+4+5+6+7+8+9+10+11+12+13+14+15+16+17+18+19+20 --local_ep 20 

# run FedEM
echo "Run FedEM"
python fedEM.py cifar10 FedEM  --n_learners 3 --n_rounds 500 --bz 32 --local_steps 20  --dataset cifar  --dirichlet_parameter  1

# run FedFac
echo "Run FedFac"
python mnist_fac2_v2.py --dirichlet_parameter 1 --rounds 500 --bz 32 --nettype pre_resnet18 --weight_decay 0.00001 --partial_fed \
 --direct_eigDecom --partial_cnn_layer 20 --local_ep 20 --threshold_p --threshold 5 --cov_thrhd 0.5


## Generalization
# FedAvg
echo "Run Generalization FedAvg"
python mnist_fac2_v2.py --dirichlet_parameter 1 --rounds 500 --nettype pre_resnet18 --weight_decay 0.00001 \
--method fed --local_ep 20 --bz 32 --new_test --test_tasks_frac 0.1 

# FedProx
echo "Run Generalization FedProx"
python mnist_fac2_v2.py --dirichlet_parameter 1 --rounds 500 --bz 32 --nettype pre_resnet18 --weight_decay 0.00001 \
--method fedprox --fedprox --mu 0.001 --local_ep 20 --new_test --test_tasks_frac 0.1 

# LG-FedAvg
echo "Run Generalization LG-FedAvg"
python mnist_fac2_v2.py --dirichlet_parameter 1 --rounds 500 --bz 32 --nettype pre_resnet18 --weight_decay 0.00001 --method lg \
--partial_cnn_layer 1+2+3+4+5+6+7+8+9+10+11+12+13+14+15+16+17+18+19+20 --local_ep 20 \
 --new_test --test_tasks_frac 0.1 --general_ensemble 

# FedEM
echo "Run Generalization FedEM"
python fedEM.py cifar10 FedEM  --n_learners 3 --n_rounds 500 --bz 32 --local_steps 20  --dataset cifar  \
 --new_test --dirichlet_parameter 1

# FedFac + LocalTrain
echo "Run Generalization FedFac + LocalTrain"
python mnist_fac2_v2.py --dirichlet_parameter 1 --rounds 500 --bz 32 --nettype pre_resnet18 --weight_decay 0.00001 --partial_fed \
 --direct_eigDecom --partial_cnn_layer 20 --local_ep 20 --threshold_p --threshold 5 --cov_thrhd 0.5 --new_test \
--test_tasks_frac 0.1 --fac_newtest_priv  train-fix

# FedFac + Ensemble
echo "Run Generalization FedFac + Ensemble"
python mnist_fac2_v2.py --dirichlet_parameter 1 --rounds 500 --bz 32 --nettype pre_resnet18 --weight_decay 0.00001 --partial_fed \
 --direct_eigDecom --partial_cnn_layer 20 --local_ep 20 --threshold_p --threshold 5 --cov_thrhd 0.5 --new_test --test_tasks_frac 0.1 \
 --general_ensemble 


## Effect of Personalized and Shared parameters
# Personalized parameters
python mnist_fac2_v2.py --dirichlet_parameter 1 --rounds 500 --bz 32 --nettype pre_resnet18 --weight_decay 0.00001 --partial_fed \
--direct_eigDecom --partial_cnn_layer 20 --local_ep 20 --threshold_p --threshold 50 --shared_remove

# Shared parameters
python mnist_fac2_v2.py --dirichlet_parameter 1 --rounds 500 --bz 32 --nettype pre_resnet18 --weight_decay 0.00001 --partial_fed \
--direct_eigDecom --partial_cnn_layer 20 --local_ep 20 --threshold_p --threshold 50 --private_remove
