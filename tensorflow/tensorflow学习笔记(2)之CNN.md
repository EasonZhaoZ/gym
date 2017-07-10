# tensorflow学习笔记(2)-CNN

参考文章：http://cs231n.github.io/convolutional-networks/

## 1.学习摘要

* Architecture Overview
* ConvNet Layers
	* Convolutional Layer
	* Pooling Layer
	* Normalization Layer
	* Fully-Connected Layer
	* Converting Fully-Connected Layers to Convolutional Layers
* ConvNet Architectures
	* Layer Patterns
	* Layer Sizing Patterns
	* Case Studies (LeNet / AlexNet / ZFNet / GoogLeNet / VGGNet)
	* Computational Considerations


## 2.学习笔记

cnn和dnn的区别，dnn模型参数爆炸太快，现在的基础设施没有办法训练，cnn通过引入图像的结构信息进行网络参数共享，最大限度减少了网络的复杂度。

![](http://cs231n.github.io/assets/nn1/neural_net2.jpeg) 
<center>图1:传统的dnn网络</center>

![](http://cs231n.github.io/assets/cnn/cnn.jpeg)
<center>图2:3D的cnn网络</center>

![](http://cs231n.github.io/assets/cnn/convnet.jpeg)
<center>图3:The activations of an example ConvNet architecture</center>

### 2.1 CONV Layer 

Conv Layer的三个重要概念: depth, stride and zero-padding. 

![](http://cs231n.github.io/assets/cnn/depthcol.jpeg)

* depth指同一个区块的filter个数：the depth of the output volume is a hyperparameter: it corresponds to the number of filters we would like to use, each learning to look for something different in the input. For example, if the first Convolutional Layer takes as input the raw image, then different neurons along the depth dimension may activate in presence of various oriented edges, or blobs of color. We will refer to a set of neurons that are all looking at the same region of the input as a depth column (some people also prefer the term fibre).
* stride指滑动窗口的间隔：we must specify the stride with which we slide the filter. When the stride is 1 then we move the filters one pixel at a time. When the stride is 2 (or uncommonly 3 or more, though this is rare in practice) then the filters jump 2 pixels at a time as we slide them around. This will produce smaller output volumes spatially.
* 窗口大小：局部特征刻画的参数，the receptive field size of the Conv Layer neurons. 
* zero-padding是对边界处理的一种方式：sometimes it will be convenient to pad the input volume with zeros around the border. The size of this zero-padding is a hyperparameter. The nice feature of zero padding is that it will allow us to control the spatial size of the output volumes (most commonly as we’ll see soon we will use it to exactly preserve the spatial size of the input volume so the input and output width and height are the same).

> zero-padding的好处：1）设计stride和窗口大小时，当stride不为1的时候，可能会出现参数对不齐的问题；2）为了处理图像边界上的细节信息；

volumn速算公式：(W−F+2P)/S+1。比如图片大小是[227\*227\*3]，第一层Conv Layer的参数：窗口大小F=11，stride S=4，zero padding P=0，那么出参的窗口大小就是(227-11+2\*0)/4 + 1 = 55，Conv Layer的depth K=96，那么output volumn的大小是[55\*55\*96]。

> Parameter Sharing的好处：控制模型复杂度的作用。Parameter sharing scheme is used in Convolutional Layers to control the number of parameters. 

Using the real-world example above, we see that there are 55\*55\*96 = 290,400 neurons in the first Conv Layer, and each has 11\*11\*3 = 363 weights and 1 bias. Together, this adds up to 290400 * 364 = 105,705,600 parameters on the first layer of the ConvNet alone. Clearly, this number is very high. It turns out that we can dramatically reduce the number of parameters by making one reasonable assumption: That if one feature is useful to compute at some spatial position (x,y), then it should also be useful to compute at a different position (x2,y2). In other words, denoting a single 2-dimensional slice of depth as a depth slice (e.g. a volume of size [55x55x96] has 96 depth slices, each of size [55x55]), we are going to constrain the neurons in each depth slice to use the same weights and bias. With this parameter sharing scheme, the first Conv Layer in our example would now have only 96 unique set of weights (one for each depth slice), for a total of 96 * 11 * 11 * 3 = 34,848 unique weights, or 34,944 parameters (+96 biases). Alternatively, all 55*55 neurons in each depth slice will now be using the same parameters. In practice during backpropagation, every neuron in the volume will compute the gradient for its weights, but these gradients will be added up across each depth slice and only update a single set of weights per slice.

> Parameter Sharing的问题，Note that sometimes the parameter sharing assumption may not make sense. This is especially the case when the input images to a ConvNet have some specific centered structure, where we should expect, for example, that completely different features should be learned on one side of the image than another. One practical example is when the input are faces that have been centered in the image. You might expect that different eye-specific or hair-specific features could (and should) be learned in different spatial locations. In that case it is common to relax the parameter sharing scheme, and instead simply call the layer a Locally-Connected Layer.

![](http://cs231n.github.io/assets/cnn/weights.jpeg)

> [55\*55\*96]第一层Conv Layer的Volumn。

### 2.2 Pooling Layer

一般会再Conv Layer之间添加Pooling Layer，控制模型复杂度来control overfitting。

> General pooling. In addition to max pooling, the pooling units can also perform other functions, such as average pooling or even L2-norm pooling. Average pooling was often used historically but has recently fallen out of favor compared to the max pooling operation, which has been shown to work better in practice.

![](http://cs231n.github.io/assets/cnn/pool.jpeg)

丢弃pooling layer也有人再尝试，Getting rid of pooling. Many people dislike the pooling operation and think that we can get away without it. For example, Striving for Simplicity: The All Convolutional Net proposes to discard the pooling layer in favor of architecture that only consists of repeated CONV layers. To reduce the size of the representation they suggest using larger stride in CONV layer once in a while. Discarding pooling layers has also been found to be important in training good generative models, such as variational autoencoders (VAEs) or generative adversarial networks (GANs). It seems likely that future architectures will feature very few to no pooling layers. 


### 2.3 Conv Architectures 

INPUT -> [[CONV -> RELU]\*N -> POOL?]\*M -> [FC -> RELU]*K -> FC

where the * indicates repetition, and the POOL? indicates an optional pooling layer. Moreover, N >= 0 (and usually N <= 3), M >= 0, K >= 0 (and usually K < 3). 

> In practice: use whatever works best on ImageNet. If you’re feeling a bit of a fatigue in thinking about the architectural decisions, you’ll be pleased to know that in 90% or more of applications you should not have to worry about these. I like to summarize this point as “don’t be a hero”: Instead of rolling your own architecture for a problem, you should look at whatever architecture currently works best on ImageNet, download a pretrained model and finetune it on your data. You should rarely ever have to train a ConvNet from scratch or design one from scratch. I also made this point at the Deep Learning school.


## 声名在外的ConvNet


* LeNet. The first successful applications of Convolutional Networks were developed by Yann LeCun in 1990’s. Of these, the best known is the LeNet architecture that was used to read zip codes, digits, etc.
* AlexNet. The first work that popularized Convolutional Networks in Computer Vision was the AlexNet, developed by Alex Krizhevsky, Ilya Sutskever and Geoff Hinton. The AlexNet was submitted to the ImageNet ILSVRC challenge in 2012 and significantly outperformed the second runner-up (top 5 error of 16% compared to runner-up with 26% error). The Network had a very similar architecture to LeNet, but was deeper, bigger, and featured Convolutional Layers stacked on top of each other (previously it was common to only have a single CONV layer always immediately followed by a POOL layer).
* ZF Net. The ILSVRC 2013 winner was a Convolutional Network from Matthew Zeiler and Rob Fergus. It became known as the ZFNet (short for Zeiler & Fergus Net). It was an improvement on AlexNet by tweaking the architecture hyperparameters, in particular by expanding the size of the middle convolutional layers and making the stride and filter size on the first layer smaller.
* GoogLeNet. The ILSVRC 2014 winner was a Convolutional Network from Szegedy et al. from Google. Its main contribution was the development of an Inception Module that dramatically reduced the number of parameters in the network (4M, compared to AlexNet with 60M). Additionally, this paper uses Average Pooling instead of Fully Connected layers at the top of the ConvNet, eliminating a large amount of parameters that do not seem to matter much. There are also several followup versions to the GoogLeNet, most recently Inception-v4.
* VGGNet. The runner-up in ILSVRC 2014 was the network from Karen Simonyan and Andrew Zisserman that became known as the VGGNet. Its main contribution was in showing that the depth of the network is a critical component for good performance. Their final best network contains 16 CONV/FC layers and, appealingly, features an extremely homogeneous architecture that only performs 3x3 convolutions and 2x2 pooling from the beginning to the end. Their pretrained model is available for plug and play use in Caffe. A downside of the VGGNet is that it is more expensive to evaluate and uses a lot more memory and parameters (140M). Most of these parameters are in the first fully connected layer, and it was since found that these FC layers can be removed with no performance downgrade, significantly reducing the number of necessary parameters.
* ResNet. Residual Network developed by Kaiming He et al. was the winner of ILSVRC 2015. It features special skip connections and a heavy use of batch normalization. The architecture is also missing fully connected layers at the end of the network. The reader is also referred to Kaiming’s presentation (video, slides), and some recent experiments that reproduce these networks in Torch. ResNets are currently by far state of the art Convolutional Neural Network models and are the default choice for using ConvNets in practice (as of May 10, 2016). In particular, also see more recent developments that tweak the original architecture from Kaiming He et al. Identity Mappings in Deep Residual Networks (published March 2016).

