import torch


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

        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            #torch.nn.BatchNorm2d(32),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.Conv2d(32, 48, kernel_size=3, stride=2, padding=1),
            #torch.nn.BatchNorm2d(48),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(negative_slope=0.2),
        )
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(64, 128),
            torch.nn.Dropout(p=0.3),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.Linear(128, 64),
            torch.nn.Dropout(p=0.3),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.Linear(64, 2),
            torch.nn.Tanh()
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
        x = observation/255.
        x = x.permute(0, 3, 1, 2)
        x = self.cnn(x)
        x = torch.mean(x, (2, 3))  # global average pooling
        x = self.mlp(x)
        #x = x * torch.Tensor([1.1, 1.]).cuda()  # eliminate steering sensitivity
        #x = torch.clip(x, -1, 1)
        return x

    def actions_to_classes(self, actions):
        """
        1.1 c)
        For a given set of actions map every action to its corresponding
        action-class representation. Every action is represented by a 1-dim vector 
        with the entry corresponding to the class number.
        actions:        python list of N torch.Tensors of size 3
        return          python list of N torch.Tensors of size 1
        """
        pass

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
        reg_output = [torch.zeros(2) for a in actions]
        for i, a in enumerate(actions):
            # add steering
            reg_output[i][0] = a[0]

            # scale gas and braking to [-1, 1]
            if a[2] > 0:
                reg_output[i][1] = -1
            elif a[1] > 0:
                reg_output[i][1] = 1

        return reg_output

    def scores_to_action(self, scores):
        """
        1.1 c) / 1.2 c)
        Maps the scores predicted by the network to an action-class and returns
        the corresponding action [steer, gas, brake].
                        C = number of classes
        scores:         python list of torch.Tensors of size 2
        return          (float, float, float)
        """
        scores = scores.cpu().detach()
        score = scores[0]
        steering = score[0].item()
        gas, brake = 0., 0.

        long_contrl = score[1].item()
        if long_contrl > 0.:
            gas = long_contrl * 0.5
            brake = 0.
        elif long_contrl < 0.:
            gas = 0.
            brake = long_contrl * 0.8
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
        steering = steer_crop.sum(dim=1, keepdim=True)
        gyro_crop = observation[:, 88, 58:86, 0].reshape(batch_size, -1)
        gyroscope = gyro_crop.sum(dim=1, keepdim=True)
        return speed, abs_sensors.reshape(batch_size, 4), steering, gyroscope
