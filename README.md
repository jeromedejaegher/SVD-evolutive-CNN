# SVD-evolutive-CNN
(Pytorch implementation)
Toy example of a tool to optimize CNN layers widths, according their singular values decomposition (SVD).

## Idea and principle
Given a NN structure, the tool performs a SVD decomposition, pruns low-energy/low variance dims, and adds new dims based on the energy distribution of the singular vectors.

## Usage :
Given a general network architecture, it optimizes layers-width during training, enabling to use networks weights from a step to the other.
Thus, it enables a **re-use of weights of a previously trained network, saving time and energy-consumption**.

## Results :
Tested on the MNIST datasets, gives a **98,5% accuracy** with a light 6-layers ResNet, with only **12.5k params** (starting from 2M params). This reduction can be reached within hours, **without loss of stability** (on this easy dataset).

## Directions to improve the model : 
- On Residual Blocks, perform svd on Id + AB instead of A and B
- The "optimize layers" util can be split in two : one utils to manage layers enlargment or layers shrinking, and another tool which layers to enlarge or to shrinking, given a constraint (GPU memory...). The "optimize layers" util has a step of 1, but this can be changed also.
- Proto, needs quite a lot of work to industrialize ;)
