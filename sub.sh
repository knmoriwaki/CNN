#!/bin/sh

data_path=./TrainingData/voxel_features_no_cut.txt
data_path_test=./TestData/voxel_features_no_cut.txt

epoch=10
batch_size=32

output_dir=./output/ep${epoch}_bs${batch_size}
mkdir -p $output_dir

norm_param_file=${output_dir}/norm_param.txt

model_args="--model_dir $output_dir --model MLP --r_drop 0.2 --loss bce --source_id 0 1 2 3 4 --target_id 5 --norm_param_file $norm_param_file "

python3 main.py --isTrain $model_args --data_path $data_path --epoch $epoch --batch_size $batch_size
#python3 main.py $model_args --data_path $data_path_test 

#data_dir_test=./PS1_PS2_Data
#python3 main.py $model_args --data_dir $data_dir_test --file_id Pk_PS1
#python3 main.py $model_args --data_dir $data_dir_test --file_id Pk_PS2