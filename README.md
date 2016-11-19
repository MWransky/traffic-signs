# Self-Driving Car Engineer Nanodegree
# Deep Learning
## Project: Build a Traffic Sign Recognition Program

## Summary:
This repo contains my submission for the traffic sign recognition project for Udacity's Self-Driving Car Engineer Nanodegree program.

- `report.html` contains the executed code from image loading to training and testing the deep neural network
- `/tranined_weights` contains the saved weights for the final network
- `/test_images` contains 10 sample 32x32 color images of signs used to challenge the trained network

## Network Performance:
The final network has training and validation accuracies of over 98% and a testing accuracy of over 92%.

## Network Architecture:
My final architecture involves a convolutional neural network (CNN) similar to that of AlexNet, but with several important updates/changes. In general, the architecture incorporates two convolution layers followed by two fully connected layers.

*General Parameters*:
- Number of hidden layers = 64
- Number of patches for the convolutions = 5
- Depth of hidden layers = 64

*1st Convolutional Layer*

The first layer is fed the 32x32x3 color image. This image is put through a 2-dimensional convolution with a stride of 1. Next, the result of the convolution is added with a bias vector and their sum is processed using the `tf.nn.relu` activation operator. Then, the result of this activation is put through a max pooling operator using kernal of `[1,2,2,1]` and a stride of `[1,2,2,1]`. Finally, the result of this max pooling is put through a local response normalization operation.

*2nd Convolutional Layer*

The second convolutional layer is identical to the first, with two main exceptions. First is that the second layer is fed the output of the first convolutional layer. Second is that following the activation layer, the local responsed normalization occurs prior to the max pooling operator.

*1st Fully Connected Layer*

The output of the second convolutional layer is reshaped and multiplied by a weight matrix. The result of this multiplication is added to a bias vector, and that summation is passed through the `tf.nn.relu` activation function.

*2nd Fully Connected Layer*

The network concludes by multiplying the result of the 1st fully connected layer with a weight matrix, adding a bias, and returning the result for the softmax probability operation to provide the final classification.

## Training Architecture
Weights used in the convolutional layers were initialized using a truncated normal distribution with a standard deviation of 0.1. Bias weights were either initialized to zeros or ones. Weights for the fully connected layers were initialized also using a truncated normal distribution with a standard deviation of 0.1.

Image classes were transformed into one-hot encodings.

The model was trained using batch sets of 64 with 4501 training steps.

A reduced mean, cross entropy loss function was fed the logits from the last fully connected layer. This loss was then minimized using a momentum optimizer with a decaying learning rate. The initial learning rate was set to 0.01 and exponentially decayed per each training step.

Regularization techniques such as imposing an L2 norm to the weights or using dropout regularization had no significant impact in improving the model's performance, so they were ultimately excluded from the final model.
