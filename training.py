import torch
import torch.nn as nn
import random
import time

import numpy as np
import matplotlib.pyplot as plt

from network import ClassificationNetwork
from demonstrations import load_demonstrations


def train(data_folder, trained_network_file):
    """
    Function for training the network.
    """
    infer_action = ClassificationNetwork()
    max_lr = 1e-3
    optimizer = torch.optim.Adam(infer_action.parameters(), lr=max_lr)
    loss_function = nn.BCELoss()

    observations, actions = load_demonstrations(data_folder)
    observations = [torch.Tensor(observation) for observation in observations]
    actions = [torch.Tensor(action) for action in actions]

    # validation=True
    # validation_size_percentage = 10
    # if validation:
    #     rand.sample(range(1, len(observations)//validation_size_percentage), len(observations)//validation_size_percentage)
    #     validation_observations = []
    #     validation_actions = []
    #     for i in range(len(observations)//validation_size_percentage):
    #         index = random.randrange(len(observations) - i - 1)
    #         validation_observations.append(observations.pop(index))
    #         validation_actions.append(infer_action.actions_to_multiclass(actions.pop(index)))


    batches = [batch for batch in zip(observations,
                                      infer_action.actions_to_multiclass(actions))]

    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    nr_epochs = 200
    batch_size = 64
    steps_per_epoch = len(batches) // batch_size
    start_time = time.time()

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                    max_lr=max_lr,
                                                    epochs=nr_epochs,
                                                    steps_per_epoch=steps_per_epoch)
    infer_action.train()

    fig = plt.figure()
    train_loss = []

    for epoch in range(nr_epochs):
        random.shuffle(batches)

        total_loss = 0
        batch_in = []
        batch_gt = []
        for batch_idx, batch in enumerate(batches):
            batch_in.append(batch[0].to(device))
            batch_gt.append(batch[1].to(device))

            if (batch_idx + 1) % batch_size == 0:  # last 39 elements are cut off and not used for batch
                batch_in = torch.reshape(torch.cat(batch_in, dim=0),
                                         (-1, 96, 96, 3))
                batch_gt = torch.reshape(torch.cat(batch_gt, dim=0),
                                         (-1, 4))

                batch_out = infer_action(batch_in)
                loss = loss_function(batch_out, batch_gt)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss

                batch_in = []
                batch_gt = []

                scheduler.step()
        
        # v_total_loss = validation(infer_action,validation_observations,validation_actions)

        # #Plot
        # train_loss.append(total_loss)
        # valid_loss.append(v_total_loss)
        # fig.close()
        # fig = plt.figure()
        # plt.plot(train_loss,np.arrange(epoch))
        # plt.plot(valid_loss,np.arrange(epoch))
        # plt.show(blocking=False)

        time_per_epoch = (time.time() - start_time) / (epoch + 1)
        time_left = (1.0 * time_per_epoch) * (nr_epochs - 1 - epoch)
        print("Epoch %5d\t[Train]\tloss: %.6f \tETA: +%fs" % (
            epoch + 1, total_loss, time_left))

    torch.save(infer_action, trained_network_file)

def validation(infer_action,validation_observations,validation_actions):
    total_loss = 0
    infer_action.eval()
    for i in range(len(validation_observations)):
        action_scores = infer_action.scores_to_action(infer_action(torch.Tensor(
                np.ascontiguousarray(validation_observations[None])).to(device)))
        loss = loss_function(batch_out, batch_gt)


#observation = np.load("/home/lenny/Uni/SDC/ex_01_imitation_learning/sdc_challenges/data/teacher/observation_1.npy")
#is_critical_state(observation)