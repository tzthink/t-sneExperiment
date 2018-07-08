import urllib2
import numpy as np
from tsne import tsne
import pylab
import os

def getDataset(url, name):
    file = urllib2.urlopen(url)
    with open(name, 'wb') as output:
        output.write(file.read())

def processData(name):
    X = open("X.txt", 'a')
    Y = open("Y.txt", 'a')
    dict = {'Iris-setosa\n': '0.0', 'Iris-versicolor\n': '1.0', 'Iris-virginica\n': '2.0'}
    with open(name, 'r+') as file:
        for line in file:
            index = line.rfind(',')
            if index < 0:
                continue
            line = line.replace(',', ' ')
            X.write(line[0:index])
            X.write('\n')
            Y.write(dict[line[index+1:]])
            Y.write('\n')



if __name__ == '__main__':
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    name = 'iris.txt'
    if not os.path.exists(name):
        getDataset(url, name)
    else:
        print ("The iris data is already existing.")

    if not os.path.exists('X.txt') or not os.path.exists('Y.txt'):
        processData(name)
    else:
        print ("The pre-processed data X.txt and Y.txt are already existing.")

    print("Run Y = tsne.tsne(X, no_dims, perplexity) to perform t-SNE on your dataset.")
    print("Running example on 150 IRIS data...")
    X = np.loadtxt("X.txt")
    labels = np.loadtxt("Y.txt")
    Y = tsne(X, 2, 50, 20.0)
    pylab.scatter(Y[:, 0], Y[:, 1], 20, labels)
    pylab.savefig('t-sne_iris.png')
    pylab.show()
