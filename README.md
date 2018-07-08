#### Introduction

t-distributed stochastic neighbor embedding (t-SNE) is a dimensionality reduction algorithm developed by Geoffrey Hinton and Laurens van der Maaten. t-SNE can embed high-dimensional data into a space of two or three dimensional, so it is often used for high-dimensional data visualization in the machine learning. 

The implementation of t-SNE can be find here. The idea in the t-SNE technique is to calculate the similarities in the input dataset X. The calculation generates a distribution of the similarity P at the end. Then, t-SNE try to use the low-dimensional (2 or 3) points’ distribution Q to mimic the distribution of real data P. t-SNE uses the Kullback-Leibler divergence as the metric to measure the difference between distribution P and distribution Q. Finally, gradient descent is used for the model training. 

I use t-SNE to visualize the iris dataset hold by UCI Machine Learning Repository. The iris dataset has 150 instances with 4 attributes. The dataset is often used for the multi-class classification problem. The visualization code for iris dataset can be found here. I train the t-SNE for 1000 iterations, and print out the training error every 10 iterations.

#### How to play
Run `python experiment.py`. The visualized image `t-sne_iris.png` will be given.