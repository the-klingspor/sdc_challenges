import numpy as np
from demonstrations import load_demonstrations
import matplotlib.pyplot as plt



[observations, actions] = load_demonstrations("/home/lenny/Uni/SDC/ex_01_imitation_learning/sdc_challenges/data/teacher_new")

for index, image in enumerate(observations):
    print(actions[index])
    plt.figure()
    plt.imshow(image) 
    plt.show()

