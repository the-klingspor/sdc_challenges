import torch
from torchvision import models
from torchvision import transforms


class ClassificationNetwork(torch.nn.Module):
    def __init__(self):
        """
        1.1 d)
        Implementation of the network layers. The image size of the input
        observations is 96x96 pixels.
        """
        super().__init__()

        # setting device on GPU if available, else CPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.to_grayscale = transforms.Grayscale(num_output_channels=1)
        self.augment = torch.nn.Sequential(
            transforms.RandomAffine(degrees=20, translate=(0.1, 0.1)),
            transforms.RandomErasing(scale=(0.02, 0.1)),
        )

        """
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.Conv2d(32, 48, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(48),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.Flatten()
        )
        """
        self.cnn = torch.nn.Sequential(*(list(models.regnet_y_400mf(pretrained=True).children())[:-1]))

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(447, 128),  # nr_cnn_output feature maps + 7 sensor values
            torch.nn.Dropout(p=0.3),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.Linear(128, 64),
            torch.nn.Dropout(p=0.3),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.Linear(64, 4, bias=False),
            torch.nn.Sigmoid()
        )

        if torch.cuda.is_available():
            self.cnn = self.cnn.cuda()
            self.mlp = self.mlp.cuda()

    def forward(self, observation):
        """
        1.1 e)
        The forward pass of the network. Returns the prediction for the given
        input observation.
        observation:   torch.Tensor of size (batch_size, 96, 96, 3)
        return         torch.Tensor of size (batch_size, C)
        """
        batch_size = observation.shape[0]
        sensor_values = self.extract_sensor_values(observation, batch_size)
        sensor_values = torch.cat(sensor_values, dim=1)[:, :, None, None]

        x = observation.permute(0, 3, 1, 2)
        #x = self.to_grayscale(x)
        if self.training:
            x = self.augment(x)
        x = x / 255.
        x = self.normalize(x)
        x = self.cnn(x)
        x = torch.cat((x, sensor_values), dim=1).squeeze()
        x = self.mlp(x)

        return x

    def actions_to_multiclass(self, actions):
        """
        1.1 c)
        For a given set of actions map every action to its corresponding
        action-class representation. Every action is represented by a 5-dim vector
        with the entry corresponding to the multiclass output. The classes
        correspond to the four possible keys "Right", "Left", "Gas" and "Brake".
        actions:        python list of N torch.Tensors of size 3
        return          python list of N torch.Tensors of size 4
        """
        multiclass = [torch.zeros(4) for _ in actions]
        for i, a in enumerate(actions):
            # add steering
            if a[0] > 0:  # right
                multiclass[i][0] = a[0]
            elif a[0] < 0:  # left
                multiclass[i][1] = -a[0]
            # add gas
            multiclass[i][2] = a[1]
            # add brake
            multiclass[i][3] = a[2]

        return multiclass

    def actions_to_regs(self, actions):
        """
        1.2 c)
        For a given set of actions map every action to its corresponding
        action-regression representation. Every action is represented by a 2-dim vector
        with the first entry being the steering and the second being the gas/
        braking.
        The first entry of the action is in {-1, 0, 1}, it can remain unchanged.
        The second is the gas in {0, 0.5} and the third braking in {0, 0.8}.
        We map them to a combined value in {-1, 0, 1}.
        actions:        python list of N torch.Tensors of size 3
        return          python list of N torch.Tensors of size 2
        """
        reg_output = [torch.zeros(2) for _ in actions]
        for i, a in enumerate(actions):
            # add steering
            reg_output[i][0] = a[0]

            # scale gas and braking to [-1, 1]
            if a[2] > 0:  # braking
                reg_output[i][1] = -1.
            elif a[1] > 0:  # gas
                reg_output[i][1] = 1.

        """
        # Control if actions are parsed and parsed back correctly
        for i in range(500):
            print(actions[i])
            print(reg_output[i])
            print(self.scores_to_action(reg_output[i][None, :]))
            print("_____")
        """

        return reg_output

    def scores_to_action(self, scores):
        """
        1.1 c) / 1.2 c)
        Maps the scores predicted by the network to an action-class and returns
        the corresponding action [steer, gas, brake].
                        C = number of classes
        scores:         python list of torch.Tensors of size 4
        return          (float, float, float)
        """
        if scores.is_cuda:
            scores = scores.cpu()
        scores = scores.detach()
        score = scores.squeeze()
        right = score[0].item()
        left = score[1].item()
        gas = score[2].item()
        brake = score[3].item()

        # Only retain max steering and max longitudinal movement
        if right > left:
            steering = right
        elif left > right:
            steering = -left
        else:
            steering = 0

        if gas > brake:
            brake = 0
        elif brake > gas:
            gas = 0
        else:
            gas, brake = 0, 0

        return steering, gas, brake

    def extract_sensor_values(self, observation, batch_size):
        """
        observation:    python list of batch_size many torch.Tensors of size
                        (96, 96, 3)
        batch_size:     int
        return          torch.Tensors of size (batch_size, 1),
                        torch.Tensors of size (batch_size, 4),
                        torch.Tensors of size (batch_size, 1),
                        torch.Tensors of size (batch_size, 1)
        """
        speed_crop = observation[:, 84:94, 12, 0].reshape(batch_size, -1)
        speed = speed_crop.sum(dim=1, keepdim=True) / 255
        abs_crop = observation[:, 84:94, 18:25:2, 2].reshape(batch_size, 10, 4)
        abs_sensors = abs_crop.sum(dim=1) / 255
        steer_crop = observation[:, 88, 38:58, 1].reshape(batch_size, -1)
        steering = steer_crop.sum(dim=1, keepdim=True) / 255
        gyro_crop = observation[:, 88, 58:86, 0].reshape(batch_size, -1)
        gyroscope = gyro_crop.sum(dim=1, keepdim=True) / 255
        return speed, abs_sensors.reshape(batch_size, 4), steering, gyroscope
