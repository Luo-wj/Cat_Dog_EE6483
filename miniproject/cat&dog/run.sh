gpu_id=1

python3 train.py --method CNN --name CNN_no_aug --batch_size 1000 \
--seed 0 --gpus $gpu_id --data_aug False
python3 train.py --method CNN --name CNN_no_aug --batch_size 1000 \
--seed 1 --gpus $gpu_id --data_aug False
python3 train.py --method CNN --name CNN_no_aug --batch_size 1000 \
--seed 2 --gpus $gpu_id --data_aug False

python3 train.py --method CNN --name CNN --batch_size 1000 \
--seed 0 --gpus $gpu_id --data_aug True
python3 train.py --method CNN --name CNN --batch_size 1000 \
--seed 1 --gpus $gpu_id --data_aug True
python3 train.py --method CNN --name CNN --batch_size 1000 \
--seed 2 --gpus $gpu_id --data_aug True



python3 train.py --method ResNet --name ResNet_no_aug --batch_size 1000 \
--seed 0 --gpus $gpu_id --data_aug False
python3 train.py --method ResNet --name ResNet_no_aug --batch_size 1000 \
--seed 1 --gpus $gpu_id --data_aug False
python3 train.py --method ResNet --name ResNet_no_aug --batch_size 1000 \
--seed 2 --gpus $gpu_id --data_aug False

python3 train.py --method ResNet --name ResNet --batch_size 1000 \
--seed 0 --gpus $gpu_id --data_aug True
python3 train.py --method ResNet --name ResNet --batch_size 1000 \
--seed 1 --gpus $gpu_id --data_aug True
python3 train.py --method ResNet --name ResNet --batch_size 1000 \
--seed 2 --gpus $gpu_id --data_aug True



python3 train.py --method ResNet_CBAM --name ResNet_CBAM_no_aug --batch_size 1000 \
--seed 0 --gpus $gpu_id --data_aug False
python3 train.py --method ResNet_CBAM --name ResNet_CBAM_no_aug --batch_size 1000 \
--seed 1 --gpus $gpu_id --data_aug False
python3 train.py --method ResNet_CBAM --name ResNet_CBAM_no_aug --batch_size 1000 \
--seed 2 --gpus $gpu_id --data_aug False

python3 train.py --method ResNet_CBAM --name ResNet_CBAM --batch_size 1000 \
--seed 0 --gpus $gpu_id --data_aug True
python3 train.py --method ResNet_CBAM --name ResNet_CBAM --batch_size 1000 \
--seed 1 --gpus $gpu_id --data_aug True
python3 train.py --method ResNet_CBAM --name ResNet_CBAM --batch_size 1000 \
--seed 2 --gpus $gpu_id --data_aug True