# 2D convolutional neural network 


## Requirement

Install the following libraries with `pip`.
- torch
- torchvision
- torchinfo
- tqdm
- matplotlib

or just use docker
## How to Run

Put your data in TrainingData and TestData

Run main.py 
```
python main.py --isTrain # for training
python main.py # for test
```

Run training and test code by 
```
./sub.sh
```
You can change the parameters in sub.sh


Use plot.ipynb to check the model performance. 

Loss function:  
![loss](image/loss.png) 

Test output:  
![test](image/test_image.png) 

Images generated during training:  
![iter](image/training_iter.png)


## References


## Known Issues


