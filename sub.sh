#!/bin/sh

n_feature=1
seq_length=45412

model=CNN 

hidden_dim=8
n_layer=4
r_drop=0.5

batch_size=64
epoch=4
lr=0.001

loss=nllloss
loss=l1norm

data_dir=./training_data
ndata=2000

test_dir=./test_data
ndata_test=10

model_base=${model}_nf${n_feature}_hd${hidden_dim}_nl${n_layer}_r${r_drop}_${loss}_bs${batch_size}_ep${epoch}_lr${lr}
output_dir=./output/${model_base}
model_dir=${output_dir}
fname_norm=./param/norm_params_nf${n_feature}_seq${seq_length}.txt

mkdir -p $model_dir

### training ###
today=`date '+%Y-%m-%d'`
python main.py --isTrain --data_dir $data_dir --ndata $ndata --model_dir $model_dir --model ${model} --fname_norm $fname_norm --n_feature $n_feature --seq_length $seq_length --hidden_dim $hidden_dim --n_layer $n_layer --r_drop $r_drop --batch_size $batch_size --epoch $epoch --lr $lr --loss $loss > ./tmp/out_${today}_${model_base}.log
echo "# output ./tmp/out_${today}_${model_base}.log"

### test ###
model_file_name=model.pth
python main.py --test_dir $test_dir --ndata $ndata_test --model_dir $model_dir --model ${model} --fname_norm $fname_norm --n_feature $n_feature --seq_length $seq_length --hidden_dim $hidden_dim --n_layer $n_layer --batch_size $batch_size --epoch $epoch --lr $lr --loss $loss

