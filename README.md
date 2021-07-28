# SVD-evolutive-CNN
(Pytorch implementation)
Toy example of a tool to optimize CNN layers widths, according their singular values decomposition (SVD).

## Idea and principle
Given a NN structure, the tool performs a SVD decomposition on each layer weight.
On the singular values diagonal matrix S :
- the tool pruns lowest (low-energy/low variance) values, and pruns dims along the corresponding vectors on matrix U and V.T, and 
- adds new dims on layer where S have high-energy values, orthogonal from existing sigular vectors.

## Usage :
Given a general network architecture, it optimizes layers-width during training, enabling to use networks weights from a step to the other.
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
