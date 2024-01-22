#!/bin/sh

for hidden_dim in 32 128
do
	for batch_size in 4 128
	do
		lr=`bc <<< "scale=5; 0.00025*$batch_size"`
		./sub.sh $hidden_dim $batch_size $lr 
	done
done


	
