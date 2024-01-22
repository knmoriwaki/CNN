# Asteroseismology

Estimate the inclination angle from the spectrum

## Requirement

- Python 3.8+

Install the following libraries with `pip`.
- torch==1.12.0
- torchvision==0.13.0
- torchinfo
- tqdm
- matplotlib

## How to Run

Put your training and validation data (e.g., 0000000.0.data, Combinations.txt) at ./training_data

Put your test data at ./test_data

Run training and test code by 
```
./sub.sh
```
You can change the parameters in run.sh.


Use plot.ipynb to check the model performance. 

Loss function:  
![loss](image/loss.png) 

Test output:  
![test](image/test_image.png) 

Images generated during training:  
![iter](image/training_iter.png)

You can check the model structure in the output file at `./tmp`


- Input shape: (batch_size, seq_length, n_feature)

- Output shape: (batch_size, output_dim)


Loss functions:

- NLLLoss: set output_dim > 1

- L1norm: set output_dim = 1


## References


## Known Issues


