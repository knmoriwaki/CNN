#!/bin/sh

data_dir=./TrainingData/dataset_700Mpc_250
data_dir_test=./TestData/dataset_700Mpc_250

epoch=10
batch_size=32

output_dir=./output/ep${epoch}_bs${batch_size}
mkdir -p $output_dir

model_args="--model_dir $output_dir --r_drop 0.2 --loss nllloss --input_dim 7 --output_dim 10"

#python3 main.py --isTrain $model_args --data_dir $data_dir --idata_start 0 --ndata 360 --n_noise 100 --epoch $epoch --batch_size $batch_size 
#python3 main.py $model_args --data_dir $data_dir_test --idata_start 360 --ndata 40

data_dir_test=./PS1_PS2_Data
python3 main.py $model_args --data_dir $data_dir_test --file_id Pk_PS1
python3 main.py $model_args --data_dir $data_dir_test --file_id Pk_PS2