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
    max_lr = 5e-4
    optimizer = torch.optim.Adam(infer_action.parameters(), lr=max_lr)
    loss_function = nn.BCELoss()

    observations, actions = load_demonstrations(data_folder)
    observations = [torch.Tensor(observation) for observation in observations]
    actions = [torch.Tensor(action) for action in actions]

    validation = False
    plot = False
    val_size = 0.2
    if validation:
        n_val = int(len(observations)*val_size)
        val_indices = np.arange(len(observations))
        np.random.shuffle(val_indices)
        observations = [observations[i] for i in val_indices]
        actions = [actions[i] for i in val_indices]

        observations_val = observations[:n_val]
        observations = observations[n_val:]

        actions_val = actions[:n_val]
        actions = actions[n_val:]

    batches = [batch for batch in zip(observations,
                                      infer_action.actions_to_multiclass(actions))]

    if validation:
        val_batches = [batch for batch in zip(observations_val,
                                              infer_action.actions_to_multiclass(actions_val))]

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

    train_loss = []
    val_loss = []

    for epoch in range(nr_epochs):
        random.shuffle(batches)

        total_loss = 0
        batch_in = []
        batch_gt = []
        for batch_idx, batch in enumerate(batches):
            batch_in.append(batch[0].to(device))
            batch_gt.append(batch[1].to(device))

            if (batch_idx + 1) % batch_size == 0:  # last elements are cut off and not used for batch
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

        train_loss.append(total_loss)

        if validation:
            val_total_loss = validation_loss(infer_action,
                                             val_batches,
                                             device,
                                             batch_size,
                                             loss_function)

            val_loss.append(val_total_loss)


        time_per_epoch = (time.time() - start_time) / (epoch + 1)
        time_left = (1.0 * time_per_epoch) * (nr_epochs - 1 - epoch)

        if validation:
            print("Epoch %5d\t[Train]\tloss: %.6f [Validation]\tloss: %.6f\tETA: +%fs" % (
                epoch + 1, total_loss, val_total_loss, time_left))
        else:
            print("Epoch %5d\t[Train]\tloss: %.6f \tETA: +%fs" % (
                  epoch + 1, total_loss, time_left))

    train_loss = [x.cpu().detach() for x in train_loss]
    val_loss = [x.cpu().detach() for x in val_loss]

    if plot:
        fig = plt.figure()
        plt.plot(np.arange(nr_epochs), train_loss, label="Train Loss", color='b')
        plt.plot(np.arange(nr_epochs), val_loss, label="Validation Loss", color='r')
        plt.legend()
        plt.show()

    torch.save(infer_action, trained_network_file)


def validation_loss(infer_action, val_batches, device, batch_size, loss_function):
    infer_action.eval()
    random.shuffle(val_batches)
    val_total_loss = 0
    val_batch_in = []
    val_batch_gt = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_batches):
            val_batch_in.append(batch[0].to(device))
            val_batch_gt.append(batch[1].to(device))

            if (batch_idx + 1) % batch_size == 0:  # last elements are cut off and not used for batch
                val_batch_in = torch.reshape(torch.cat(val_batch_in, dim=0),
                                             (-1, 96, 96, 3))
                val_batch_gt = torch.reshape(torch.cat(val_batch_gt, dim=0),
                                             (-1, 4))

                val_batch_out = infer_action(val_batch_in)
                loss = loss_function(val_batch_out, val_batch_gt)

                val_total_loss += loss

                val_batch_in = []
                val_batch_gt = []

    infer_action.train()
    return val_total_loss

#observation = np.load("/home/lenny/Uni/SDC/ex_01_imitation_learning/sdc_challenges/data/teacher/observation_1.npy")
#is_critical_state(observation)