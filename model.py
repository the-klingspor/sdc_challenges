import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt


class DQN(nn.Module):
    def __init__(self, action_size, device):
        """ Create Q-network
        Parameters
        ----------
        action_size: int
            number of actions
        device: torch.device
            device on which to the model will be allocated
        """
        super().__init__()

        self.device = device 
        self.action_size = action_size
        self.regularization = 1e-6
        self.to_grayscale = transforms.Grayscale(num_output_channels=1)

        #architecture: 
        self.nn = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, kernel_size=7, stride=4, padding=0),  # output: 23x23x8, kernelsize 8? with 7 you drop one column, but who cares right, nothing of interest at the boarder?
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, stride=2,padding=1),                  # output: 12x12x8
            torch.nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=0), # output: 10x10x16
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, stride=2),                            # output: 5x5x16
            torch.nn.Flatten(),
            torch.nn.Linear(400, 400),
            torch.nn.ReLU(),
            torch.nn.Linear(400, 5)
        )

        if torch.cuda.is_available():
            self.nn = self.nn.cuda()

    def forward(self, observation):
        """ Forward pass to compute Q-values
        Parameters
        ----------
        observation: np.array
            array of state(s)
        Returns
        ----------
        torch.Tensor
            Q-values  
        """
        x = observation.permute(0, 3, 1, 2)
        x = self.preprocess(x)
        x = self.nn(x)
        return x

    def extract_sensor_values(self, observation, batch_size):
        """ Extract numeric sensor values from state pixels
        Parameters
        ----------
        observation: list
            python list of batch_size many torch.Tensors of size (96, 96, 3)
        batch_size: int
            size of the batch
        Returns
        ----------
        torch.Tensors of size (batch_size, 1),
        torch.Tensors of size (batch_size, 4),
        torch.Tensors of size (batch_size, 1),
        torch.Tensors of size (batch_size, 1)
            Extracted numerical values
        """
        speed_crop = observation[:, 84:94, 13, 0].reshape(batch_size, -1)
        speed = (speed_crop== 255).sum(dim=1, keepdim=True) 
        abs_crop = observation[:, 84:94, 18:25:2, 2].reshape(batch_size, 10, 4)
        abs_sensors = (abs_crop== 255).sum(dim=1) 
        steer_crop = observation[:, 88, 38:58, 1].reshape(batch_size, -1)
        steering = (steer_crop== 255).sum(dim=1, keepdim=True)
        gyro_crop = observation[:, 88, 58:86, 0].reshape(batch_size, -1)
        gyroscope = (gyro_crop== 255).sum(dim=1, keepdim=True)
        
        return speed, abs_sensors.reshape(batch_size, 4), steering, gyroscope

    def preprocess(self, observation):
        #observation = observation.astype(np.uint8)
        observation_gray = self.to_grayscale(observation)
        # observation_gray[abs(observation_gray - 0.60116) < 0.1] = 1
        observation_gray[:,:,84:95,0:12] = 0
        observation_gray[abs(observation_gray - 0.68616) < 0.0001] = 1
        observation_gray[abs(observation_gray - 0.75630) < 0.0001] = 1
        #uncomment to see pre processed image
        plt.imshow(observation_gray, cmap='gray')
        plt.show()

        #Set values between -1 and 1 for input normalization
        return 2 * observation_gray - 1