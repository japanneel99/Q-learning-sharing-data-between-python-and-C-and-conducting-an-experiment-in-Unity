from os import stat_result
import socket
import time
import json
from matplotlib.pyplot import xcorr
import numpy as np
import re
from multiprocessing import set_start_method
from matplotlib.animation import FuncAnimation
from numpy.core import function_base
from numpy.core.arrayprint import dtype_is_implied
from tensorforce.environments import Environment
from tensorforce.agents import DeepQNetwork
import math
import matplotlib.pyplot as plt
import csv


class CustomEnvironment(Environment):
    def __init__(self):
        host = "127.0.0.1"
        port = 25001
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((host, port))

    def states(self):
        return dict(type='float', shape=(12,))

    # def actions(self):
    #     return dict(type='float', min_value=0, max_value=1.6)

    def get_state_from_unity(self):
        self.sock.sendall("GET_WHEELCHAIR_POSITION".encode("UTF-8"))
        receivedData = self.sock.recv(1024).decode("UTF-8")

        wheelchair_position_dict = json.loads(receivedData)
        wheelchair_position = [wheelchair_position_dict['x'],
                               wheelchair_position_dict['y'], wheelchair_position_dict['z']]

        self.sock.sendall("GET_PEDESTRIAN_POSITION".encode("UTF-8"))
        receivedData = self.sock.recv(1024).decode("UTF-8")
        pedestrian_pos_dict = json.loads(receivedData)
        pedestrian_position = [pedestrian_pos_dict['x'],
                               pedestrian_pos_dict['y'], pedestrian_pos_dict['z']]

        self.sock.sendall("GET_WHEELCHAIR_VELOCITY".encode("UTF-8"))
        receivedData = self.sock.recv(1024).decode("UTF-8")
        wheelchair_vel_dict = json.loads(receivedData)
        wheelchair_velocity = [wheelchair_vel_dict['x'],
                               wheelchair_vel_dict['y'], wheelchair_vel_dict['z']]

        self.sock.sendall("GET_PEDESTRIAN_VELOCITY".encode("UTF-8"))
        receivedData = self.sock.recv(1024).decode("UTF-8")
        pedestrian_vel_dict = json.loads(receivedData)
        pedestrian_velocity = [pedestrian_vel_dict['x'],
                               pedestrian_vel_dict['y'], pedestrian_vel_dict['z']]

        self.sock.sendall("GET_FEELING_REWARD".encode("UTF-8"))
        receivedData = self.sock.recv(1024).decode("UTf-8")
        feeling = json.loads(receivedData)['info']
        print(receivedData)

        return wheelchair_position, pedestrian_position, wheelchair_velocity, pedestrian_velocity, feeling

    def reset(self):
        self.sock.sendall("RESET".encode("UTF-8"))
        receivedData = self.sock.recv(1024).decode("UTF-8")

        wheelchair_position, pedestrian_position, wheelchair_velocity, pedestrian_velocity, feeling =\
            self.get_state_from_unity()
        self.state = np.array([wheelchair_position[0], wheelchair_position[1], wheelchair_position[2],
                               wheelchair_velocity[0], wheelchair_velocity[1], wheelchair_velocity[2],
                               pedestrian_position[0], pedestrian_position[1], pedestrian_position[2],
                               pedestrian_velocity[0], pedestrian_velocity[1], pedestrian_velocity[2]])

        return self.state

    def execute(self, actions):
        wheelchair_position, pedestrian_position, wheelchair_velocity, pedestrian_velocity, feeling =\
            self.get_state_from_unity()

        terminal = wheelchair_position[2] > pedestrian_position[2] + 2
        r = math.sqrt((wheelchair_position[2] - pedestrian_position[2]) ** 2 +
                      (wheelchair_position[0] - pedestrian_position[0]) ** 2)

        reward = feeling + actions - 1/r

        self.state = np.array([wheelchair_position[0], wheelchair_position[1], wheelchair_position[2],
                               wheelchair_velocity[0], wheelchair_velocity[1], wheelchair_velocity[2],
                               pedestrian_position[0], pedestrian_position[1], pedestrian_position[2],
                               pedestrian_velocity[0], pedestrian_velocity[1], pedestrian_velocity[2]])

        return self.state, terminal, reward


def main():

    dt = 0.1

    environment = CustomEnvironment()

    agent = DeepQNetwork.create(
        agent='tensorforce', environment=environment, update=64,
        optimizer=dict(optimizer='adam', learning_rate=1e-3),
        objective='policy_gradient', reward_estimation=dict(horizon=20))

    n_epoch = 10
    for i in range(n_epoch):
        states = environment.reset()
        terminal = False
        print("%d th learning....." % i)
        Wheelchair_position = []
        Pedestrian_position = []
        action_history = []
        rewards = []

        cnt_step = 0

        while not terminal:
            cnt_step += 1
            print("step: ", cnt_step)

            actions = agent.act(states=states)
            states, terminal, reward = environment.execute(actions=actions)
            agent.observe(terminal=terminal, reward=reward)
            time.sleep(0.1)

            if i == n_epoch - 1:
                Wheelchair_position.append([states[0], states[1], states[2]])
                Pedestrian_position.append([states[6], states[7], states[8]])
                action_history.append(actions)
                rewards.append(reward)

        print("number of steps", cnt_step)

    Wheelchair_position = np.array(Wheelchair_position)
    Pedestrian_position = np.array(Pedestrian_position)

    print("Pedestrian_Position: ", Pedestrian_position)
    print("Wheelchair_Position: ", Wheelchair_position)
    print("Action History: ", action_history)
    print("Rewards: ", rewards)

    plt.subplot(224)
    plt.title("Wheelchair Position vs Pedestrian Position")
    plt.xlabel("x Position")
    plt.ylabel("z Position")
    plt.scatter(
        Wheelchair_position[:, 0], Wheelchair_position[:, 2], label="Wheelchair_Position")
    plt.scatter(
        Pedestrian_position[:, 0], Pedestrian_position[:, 2], label="Pedestrian_Position")
    plt.legend()
    # plt.show()

    plt.subplot(221)
    plt.title("Pedestrian and Wheelchair x position vs  epoch = 20")
    plt.plot(Wheelchair_position[:, 0], label="Wheelchair x position")
    plt.plot(Pedestrian_position[:, 0], label="Pedestrian x position")
    plt.xlabel("time step")
    plt.ylabel("x position")
    plt.legend()

    # ######### plot action history ##########
    plt.subplot(223)
    plt.plot(action_history)
    plt.ylim([0, 2])
    plt.title("Wheelchair Velocity, epoch = 20")
    plt.xlabel("time step")
    plt.ylabel("Wheelchair Velocity")
    # plt.show()

    # ######## plot rewards ####
    plt.subplot(222)
    plt.plot(rewards)
    plt.title("Rewards distribution after learning e_poch20")
    plt.xlabel("time step")
    plt.ylabel("reward")
    plt.show()

    actionString = "SEND_ACTION, 256"
    print(actionString)
    print("action history: ", action_history)

    fig = plt.figure()
    plt.title(
        "Dynamic Graph showing the position of the wheelchair and position epoch = 20")
    plt.xlabel("x position")
    plt.ylabel("z position")

    def animate(i, Wheelchair_position, Pedestrian_position):
        print("drawing (%d/%d)" % (i, len(Wheelchair_position)))
        if i != 0:
            plt.cla()
        plt.xlim(0, 15)
        plt.ylim(-12, 5)
        plt.scatter(
            Wheelchair_position[i, 0], Wheelchair_position[i, 2], label="Wheelchair_Position")
        plt.scatter(
            Pedestrian_position[i, 0], Pedestrian_position[i, 2], label="Pedestrian_Position")

    animation = FuncAnimation(fig, animate, frames=len(Wheelchair_position),
                              interval=100, fargs=(Wheelchair_position, Pedestrian_position))

    plt.legend()
    animation.save("DynamicPlot_epoch20.png", writer="pillow")

############ Export to csv #################################

    with open('csv_data20.csv', 'w', newline='') as csvfile:
        fieldnames = ['time_step',
                      'Wheelchair_Velocity']

        thewriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
        thewriter.writeheader()  # writes the header.

        for i in range(len(action_history)):
            thewriter.writerow(
                {'time_step': i, 'Wheelchair_Velocity': action_history[i]})

    with open('csv_data_rewards20.csv', 'w', newline='') as csvfile:
        fieldnames = ['time_step', 'Rewards']

        thewriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
        thewriter.writeheader()

        for i in range(len(rewards)):
            thewriter.writerow({'time_step': i, 'Rewards': rewards[i]})

    np.savetxt("data3.csv", Pedestrian_position, delimiter=",",
               header="pedPos_x, pedPos_y, pedPos_z", comments='')

    np.savetxt("data4.csv", Wheelchair_position, delimiter=",",
               header="wheelPos_x, wheelPos_y, wheelPos_z", comments='')


################### Sending to unity ###############
    while True:
        print("Sending Data", actionString)
        states.sock.sendall(actionString.encode("UTF-8"))

        receivedData = states.recv(1024).decode("UTF-8")
        print("Received Data", receivedData)

        while True:
            if input("Continue?(y/n)").strip() == "y":
                break


main()
