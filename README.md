# SVD-evolutive-CNN
Toy example of a tool to optimize CNN layers widths, according their SVD decomposition.

Given a NN structure, the tool performs a SVD decompozition, pruns low-energy/low variance dims, and adds new dims based on the energy distribution of the singular vectors.

Tested on the MNIST datasets, gives a 98,5% accuracy with a light 6-layers ResNet, with only 12.5k params (starting from 2M params).

Directions to improve the model : 
- On Residual Blocks, perform svd on Id + AB instead of A and B
- Proto, needs quite a lot of work to industrialize ;)
