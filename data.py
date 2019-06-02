import numpy as np
import os
class data():
    Label = []
    Image = []
    count = 0
    begin = end = 0
    def __init__(self):
        self.Label = np.load("./data/label.npy")
        self.Image = np.load("./data/image.npy")
        self.count = len(self.Image)
    
    def next_batch(self, BATCH_SIZE):
        self.end = self.begin + BATCH_SIZE
        if (self.end > self.count):
            self.begin = self.end - self.count
            self.end = self.begin + BATCH_SIZE
        image = self.Image[self.begin:self.end]
        label = self.Label[self.begin:self.end]
        self.begin = self.end
        return image, label
