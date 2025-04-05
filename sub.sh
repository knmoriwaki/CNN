#!/bin/sh

data_dir=./TrainingData/dataset_700Mpc_250
data_dir_test=./TestData/dataset_700Mpc_250

epoch=100
batch_size=32

output_dir=./output/ep${epoch}_bs${batch_size}
mkdir -p $output_dir


#python main.py --isTrain --model_dir $output_dir --data_dir $data_dir --idata_start 0 --ndata 360 --input_dim 7 --epoch $epoch --batch_size $batch_size

python3 main.py --isTrain --model_dir $output_dir --data_dir $data_dir --idata_start 0 --ndata 360 --input_dim 7 --epoch $epoch --batch_size $batch_size --r_drop 0.2 --loss nllloss --output_dim 20

#python3 main.py --model_dir $output_dir --data_dir $data_dir_test --idata_start 360 --ndata 40  --input_dim 7