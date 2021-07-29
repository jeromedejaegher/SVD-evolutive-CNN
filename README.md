# SVD-evolutive-CNN
(Pytorch implementation)

Toy example of a tool to optimize neural network layers widths during training, according their singular values decomposition (SVD).
The neural network grows if the task is too difficult for the current structure, and shrinks if it is overparametrized for the task.

*Layers considered : convolution, dense, Residual Block.*

This tool can surely be extended to transformers, as a generalization of [Collaborative Attention](https://arxiv.org/abs/2006.16362)


## Idea and principle
Given a neural network structure, the tool performs a SVD decomposition on each layer weight.

On the singular values diagonal matrix S :
- the tool pruns lowest (low-energy/low variance) values, and pruns dims along the corresponding vectors on matrix U and V.T, and 
- adds new dims on layer where S have high-energy values, orthogonal from existing singular vectors.

Formally, given a layer l, an input X, an output Y and a transformation Φ : X -> y = σ (A @ X +b) on this layer, the SVD transform the matrix A as :

A = U @ Σ @ V.T, where U and V are unitary (U @ U.T = U.T @ U = Id) and and &Sigma; is diagonal.

If A is in R<sup>d<sub>out</sub> x d<sub>in</sub></sup>, and d<sub>out</sub> < d<sub>in</sub>, then &Sigma; is in R<sup> d<sub>out</sub>x d<sub>out</sub> </sup> .

We class the values of &Sigma; in decreasing order. Replacing the last one by 0 define an approximate matrix &Sigma; <sub>d<sub>out</sub> - 1</sub> .

The difference is bounded and small, and project A on a subspace of dimension d<sub>out(l)</sub> - 1</sub>. Moreover, the projection of this approximation on the first d<sub>out(l)</sub> - 1</sub> dimensions is also of dimension d<sub>out(l)</sub> - 1</sub>.
Thus, without loss of the interpretation power of the neural network, we can restrict the output of this layer in dimension d<sub>out(l)</sub> - 1</sub>, and thus reduce the number of parameters.

Now, A = U<sub>d<sub>out</sub> - 1</sub> @ Σ<sub>d<sub>out</sub> - 1</sub> @ V<sub>d<sub>out</sub> - 1, d<sub>in</sub> - 1</sub>.T is the new &Phi; (x) on this new output space. 
 We use this new output space as the input space of the next layer, setting A(l+1) = A<sub>d<sub>out(l)</sub> - 1</sub>, d<sub>out(l + 1)</sub></sub> thus reducing the size of the 2 layers and the total number of parameters of the networks.

Symetrically, on layers where singular values are high, we can expand the output space R<sup>d<sub>out</sub></sup> -> R<sup>d<sub>out</sub> + 1</sup>, allowing the neural network to find new relevant features to improve its overall accuracy.

## Usage :
Given a general network architecture, it optimizes layers-width **during training**, enabling to use networks weights from a step to the other.
Thus, it enables a **re-use of weights of a previously trained network, saving time and energy-consumption**.

## Results :
Tested on the MNIST datasets, gives a **98,5% accuracy** with a light 6-layers ResNet, with only **12.5k params** (starting from 2M params). This reduction can be reached within an hour on domestic GPU, **automatically and without loss of stability** (on this easy dataset).

Tested on the MNIST datasets, gives a **90 % accuracy** with a light 6-layers ResNet, with **250k params** (starting from ~500k params). This reduction can be reached within an hour on domestic GPU, **automatically and without loss of stability**. Given the longest training time, a more dynamic optimizing strategy (more than one dim at a time) would be a better option to avoid overfitting.

## Directions to improve the model : 
- On Residual Blocks, perform svd on Id + AB instead of A and B
- The "optimize layers" util can be split in two : one utils to manage layers enlargment or layers shrinking, and another tool which layers to enlarge or to shrinking, given a constraint (GPU memory...). The "optimize layers" util has a step of 1, but this can be changed also.
- Proto, needs quite a lot of work to industrialize ;)


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

_Jérome Dejaegher, SVD evolutive CNN, published the 29/07/2021 on GitHub_
