import Interpreter as Itp
from matplotlib import pyplot as plt
import numpy as np


def from_file(filename='InfoMap.npy'):
    return np.load(filename)


if __name__ == '__main__':
    Map = from_file()
    print(len(Map), len(Map[0]))
