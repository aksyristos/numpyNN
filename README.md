# Neural Network Classifier using numpy
Fully Connected Classifier (28 * 28) * 300 * 1 from scratch

Differentiates MNIST Zeroes from Eights and prints the margin of error through Bayes' theorem

The same classifier was implimented with 3 different loss functions:

1)Cross-Entropy

2)Hinge

3)Exponential

# Initial testing
Before using MNIST, the network (then 2 * 20 * 1) was tested with data ( belonging to vecto X = [x1, x2] ) following these two hypotheses:

H0: x1, x2 are independent with probability density funtion f0(x1, x2) = f0(x1) * f0(x2), f0 ~ N(0, 1)

H1: x1, x2 are independent with probability density funtion f1(x1, x2) = f1(x1) * f1(x2), f1 ~ 0.5 { N(-1, 1) + N(1, 1) }

These hypotheses are what are used for the calculation of the margin of error
