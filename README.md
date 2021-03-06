# SVD-evolutive-CNN
(Pytorch implementation, done in July 2021)

Toy example of a tool to **optimize neural network layers dimensions during training**, according their singular values decomposition (SVD).
The neural network grows if the task is too difficult for the current structure, and shrinks if it is overparametrized for the task.

*Layers considered : convolution, dense, Residual Block.*

This tool can surely be extended to transformers, as a generalization of [Collaborative Attention](https://arxiv.org/abs/2006.16362)

## Usage :
Given a general network architecture, it optimizes (=modifies) layers-width **during training, without loss of accuracy**, enabling to use networks weights from a step to the other.
Thus, it enables a **re-use of weights of a previously trained network, saving time and energy-consumption**.

## Results :
Tested on the MNIST dataset, gives a **98,5% accuracy** with a light 6-layers ResNet, with only **15k params** (starting from 2M params). This reduction can be reached within an hour on domestic GPU, **automatically and without loss of stability** (on this easy dataset). Approx. 50 automatic optimization steps to reduce the network by 99% without loss in accuracy.

Tested on the Fashion-MNIST dataset, gives a **90 % accuracy** with a light 6-layers ResNet, with **350k params** (starting from ~500k params). This reduction can be reached within an hour on domestic GPU, **automatically and without loss of stability**. 

Tested on the CIFAR-10 dataset, gives a **92% accuracy on train set with 2M parameters** with a 9 layers ResNet, wich is far too deep for a 32x32 images set. The test accuracy is not that good, around 75%, but I guess it is because the model has no batch norm, the learning rate is not fine-tuned and data is not augmented.
EDIT 1 : With basic data augmentation, the test accuracy increases to 85%
EDIT 2 : Applying SVD reduction to Id + AB on each resblock makes larger model. The work is still in progress, maybe thresholds for pruning or expanding the network are too low ?

## Idea and principle
Given a neural network structure, the tool performs a SVD decomposition on each layer weight.

On the singular values diagonal matrix S :
- the tool pruns lowest (low-energy/low variance) values, and pruns dims along the corresponding vectors on matrix U and V.T, and 
- adds new dims on layer where S have high-energy values, orthogonal from existing singular vectors.

Formally, given a layer l, an input X, an output Y and a transformation ?? : X -> y = ?? (A @ X +b) on this layer, the SVD transform the matrix A as :

A = U @ ?? @ V.T, where U and V are unitary (U @ U.T = U.T @ U = Id) and and &Sigma; is diagonal.

If A is in R<sup>d<sub>out</sub> x d<sub>in</sub></sup>, and d<sub>out</sub> < d<sub>in</sub>, then &Sigma; is in R<sup> d<sub>out</sub>x d<sub>out</sub> </sup> .

We class the values of &Sigma; in decreasing order. Replacing the last one by 0 define an approximate matrix &Sigma; <sub>d<sub>out</sub> - 1</sub> .

The difference is bounded and small, and projects A on a subspace of dimension d<sub>out</sub> - 1. Moreover, the projection of this approximation on the first d<sub>out</sub> - 1</sub> dimensions is also of dimension d<sub>out</sub> - 1.
Thus, without loss of the interpretation power of the neural network, we can restrict the output of this layer in dimension d<sub>out</sub> - 1, and thus reduce the number of parameters.

Now, A = U<sub>d<sub>out</sub> - 1</sub> @ ??<sub>d<sub>out</sub> - 1</sub> @ V<sub>d<sub>out</sub> - 1, d<sub>in</sub></sub>.T is the new &Phi; (x) on this new output space. 
 We use this new output space as the input space of the next layer, setting A(l+1) = A<sub>d<sub>out(l)</sub> - 1</sub>, d<sub>out(l + 1)</sub></sub> thus reducing the size of the 2 layers and the total number of parameters of the networks.
 
On following layer, the reducted weights are calculated like this : 
 
![next_layer_shrinking_equations](img/Eqn4_next_layer_approx.svg). 

Regarding bias, they are computed as the vector that minimizes :

![bias_approx_equations](img/Eqn5_bias_approx.svg). 

Symetrically, on layers where singular values are high, we can expand the output space R<sup>d<sub>out</sub></sup> -> R<sup>d<sub>out</sub> + 1</sup> with vectors orthogonal to the original output space, allowing the neural network to find new relevant features to improve its overall accuracy.


## Experimental findings to be explained : 
- in Resnet Blocks, the intermediate channel size seems to converge to a size significantly (around 3 times) smaller than input and output sizes. As if the Neural Network distillate channel information throught space, and re-channelize it before performing the addition with the (space-oriented) residue branch


## Directions to improve the model : 
- The "optimize layers" util can be split in two : one utils to manage layers enlargment or layers shrinking, and another tool which layers to enlarge or to shrinking, given a constraint (GPU memory...). 
- To be tested with transformers, batchnorm, separable convolutions...
- On ResBlock, my current intuition is that singular values of (Id + A @ B) indicates optimal width of layers, and singular values of A and B indicates optimal depth of the network. To be investigated...
- - Proto, needs quite a lot of work to industrialize ;)


## Related works : 
During my search on related works, I found these articles about neural networks and SVD.

- [Learning Low-rank Deep Neural Networks via Singular Vector Orthogonality Regularization and Singular Value Sparsification
Huanrui Yang, Minxue Tang, Wei Wen, Feng Yan, Daniel Hu, Ang Li, Hai Li, Yiran Chen](https://arxiv.org/abs/2004.09031)

- [Spectral Pruning: Compressing Deep Neural Networks via Spectral Analysis and its Generalization Error
Taiji Suzuki, Hiroshi Abe, Tomoya Murata, Shingo Horiuchi, Kotaro Ito, Tokuma Wachi, So Hirai, Masatoshi Yukishima, Tomoaki Nishimura](https://arxiv.org/abs/1808.08558)

And this hint that Transformers too can be compressed efficiently :
- [Multi-Head Attention: Collaborate Instead of Concatenate 
Jean-Baptiste Cordonnier, Andreas Loukas, Martin Jaggi](https://arxiv.org/abs/2006.16362)

## To re-use this work :

Please notice your interest through the "issues" on this repository, follow the rules of the given Licence and cite me as author as :

_J??rome Dejaegher, SVD evolutive CNN, published the 29/07/2021 on GitHub_
