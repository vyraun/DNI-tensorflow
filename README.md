# Image classification with synthetic gradient in tensorflow
I implement the ***[Decoupled Neural Interfaces using Synthetic Gradients](http://arxiv.org/abs/1608.05343)*** in tensorflow. The paper use synthetic gradient to decouple the layers in the network. This is pretty interesting since we won't suffer from **update lock** anymore. I test my model in cifar10 and archieve similar result as the paper claimed.

## Requirement
- Tensorflow, follow the [official installation](https://www.tensorflow.org/versions/r0.10/get_started/os_setup.html#download-and-setup)
- python 2.7
- CIFAR10  dataset, go to the [dataset website](https://www.cs.toronto.edu/~kriz/cifar.html)

## TODO
- apply cDNI
- apply to some more complicated network to see if it's general

## What's synthetic gradients?
We ofter optimize NN by backpropogation, which is usually implemented in some well-known framework. However, is there another way for the layers in NN to communicate with other layers? Here comes the ***synthetic gradients***! It gives us a way to allow neural networks to communicate, to learn to send messages between themselves, in a decoupled, scalable manner paving the way for multiple neural networks to communicate with each other or improving the long term temporal dependency of recurrent networks.   
The neuron in each layer will automatically produces an error signal(***δa_head***) from synthetic-layers and do the optimzation. And how did the error signal generated? Actually, the network still does the backpropogation. While the error signal(***δa***) from the objective function is not used to optimize the neuron in the network, it is used to optimize the error signal(***δa_head***) produced by the synthetic-layer. The following is the illustration from the paper:
![](https://github.com/andrewliao11/DNI-tensorflow/blob/master/misc/dni_illustration.png?raw=true)   

## Usage 
Right now I just implement the FCN version, which is set as the default network structure   
You can define some variable in command line: ex: ```python main.py -- max_step 100000 --checkpoint_dir ./model```   
```
max_step = 50000
model_name = mlp                  # the ckpt will save in $checkpoint_path/$model_name/checkpoint-*
checkpoint_dir = './checkpoint'   # the checkpint directory
gpu_fraction = 1/2 # you can define the gpu memory usage
batch_size = 256
hidden_size = 1000             	  # hidden size of the mlp
test_per_iter = 50
optim_type = adam
synthetic = False                 # ues synthetic gradient or not	
```

## Experiment Result
DNI-mlp test on cifar10     

| cls loss  | synthetic_grad loss|
|---|---|
|![](https://github.com/andrewliao11/DNI-tensorflow/blob/master/misc/dni_mlp_cls_loss.png?raw=true) |![](https://github.com/andrewliao11/DNI-tensorflow/blob/master/misc/dni_mlp_syn_loss.png?raw=true)|

| test acc  | train acc|
|---|---|
|![](https://github.com/andrewliao11/DNI-tensorflow/blob/master/misc/dni_mlp_test_acc.png?raw=true) |![](https://github.com/andrewliao11/DNI-tensorflow/blob/master/misc/dni_mlp_train_acc.png?raw=true)|

## Something Beautiful in Tensorflow
Tensorflow is known for the convenience of auto-gradient, while at the same time many people don't know how it do the backprop or calculate the backprop. And compared to Torch, there's no obvious way to access the gradOutput, gradInput. Actually, Tensorflow contain some beautiful function that makes it easier and more flexible.   
Sometimes, you might want to calculate gradient dy/dx:   
Use ```tf.gradients(y,x)```. It's very simple
If you want to calculate the gradientm given the gradient backprop from the loss, or sth you've defined (dy/dx = dy/du*du/dx, given dy/du):
Use ```tf.gradients(y,x,dy/du)```.

## Reference
- Deepmind's [post](https://deepmind.com/blog/decoupled-neural-networks-using-synthetic-gradients/) on Decoupled Neural Interfaces Using Synthetic Gradients

