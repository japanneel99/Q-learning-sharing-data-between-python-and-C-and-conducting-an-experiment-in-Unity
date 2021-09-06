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
from tensorforce import Agent
import math
import matplotlib.pyplot as plt


############################## Running the q learning library ###########################
class CustomEnvironment(Environment):
    def __init__(self, dt, pos_rx, pos_ry, pos_rz, vel_rx, vel_ry, vel_rz):
        super().__init__()

        self.dt = dt
        self.init_pos_rx = pos_rx
        self.init_pos_ry = pos_ry
        self.init_pos_rz = pos_rz
        self.init_vel_rx = vel_rx
        self.init_vel_ry = vel_ry
        self.init_vel_rz = vel_rz

    def states(self):
        return dict(type='float', shape=(6,))

    def actions(self):
        return dict(type='float', min_value=0, max_value=1.6)

    def close(self):
        super().close()

    def reset(self):
        self.state = np.array(
            [self.init_pos_rx, self.init_pos_ry, self.init_pos_rz,
             self.init_vel_rx, self.init_vel_ry, self.init_vel_rz])

        return self.state

    def execute(self, actions):
        new_pos_rx = self.state[3] * self.dt + self.state[0]
        new_pos_ry = self.state[4] * self.dt + self.state[1]
        new_pos_rz = actions * self.dt + self.state[3]

        terminal = new_pos_rx > 7
        rewards = 100
        self.state = np.array([new_pos_rx, new_pos_ry, new_pos_rz,
                               self.init_vel_rx, self.init_vel_ry, self.init_vel_rz])
        return self.state, terminal, rewards


def main():
    #################### C#からデータを取得する。Getting the data from C# #######################
    host = "127.0.0.1"
    port = 25001

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))

    # # time.sleep(0.5) recieved data1 = position info, recieveddata2 = velocity info.
    # recievedData2 = sock.recv(1024).decode("UTF-8")

    # print(recievedData1)
    # print(recievedData2)

    sock.sendall("GET_POSITION".encode("UTF-8"))
    receivedData = sock.recv(1024).decode("UTF-8")
    position_dict = json.loads(receivedData)
    position = [position_dict['x'], position_dict['y'], position_dict['z']]

    sock.sendall("GET_VELOCITY".encode("UTF-8"))
    receivedData = sock.recv(1024).decode("UTF-8")
    velocity_dict = json.loads(receivedData)
    velocity = [velocity_dict['x'], velocity_dict['y'], velocity_dict['z']]

    # json_dict2 = json.loads(recievedData2)

    # velocity = [json_dict2['x'], json_dict2['y'], json_dict['y']] this also doesnt work!!

    print(position)
    print(velocity)

    pos_rx = position[0]
    pos_ry = position[1]
    pos_rz = position[2]
    vel_rx = velocity[0]
    vel_ry = velocity[1]
    vel_rz = velocity[2]

    dt = 0.05

    environment = CustomEnvironment(
        dt, pos_rx, pos_ry, pos_rz, vel_rx, vel_ry, vel_rz)

    agent = Agent.create(
        agent='tensorforce', environment=environment, update=64,
        optimizer=dict(optimizer='adam', learning_rate=1e-3),
        objective='policy_gradient', reward_estimation=dict(horizon=20)
    )

    # q-learning
    print("test")
    n_epoch = 50
    for i in range(n_epoch):
        states = environment.reset()
        terminal = False
        print("%d th learning....." % i)
        cube_position = []
        action_history = []

        while not terminal:
            actions = agent.act(states=states)
            states, terminal, reward = environment.execute(actions=actions)
            agent.observe(terminal=terminal, reward=reward)

            if i == n_epoch - 1:
                cube_position.append([states[0], states[1], states[2]])
                action_history.append(actions)

    cube_position = np.array(cube_position)
    plt.scatter(cube_position[:, 0], cube_position[:, 1], cube_position[:, 2])
    plt.show()
    plt.plot(action_history)
    plt.show()

    # #################################### sending the action to C# ############################
    # converting vector3 to string e. 0,0,0
    actionString = "SEND_ACTION, 256"
    print(actionString)
    print("action history: ", action_history)

    # animated figure
    #fig = plt.figure()

    # def animate(i, cube_position):
    #   if i != 0:
    #     plt.cla()
    # plt.xlim(0, 10)
    #plt.ylim(0, 10)
    # plt.scatter(cube_position[i, 0],
    #           cube_position[i, 1], cube_position[i, 2])

    # animation = FuncAnimation(fig, animate, frames=len(
    #   cube_position), interval=100, fargs=(cube_position))

    #animation.save("b.png", writer="pillow")

    # converting string to byte and sending to C#
    while True:
        print("sending data", actionString)
        sock.sendall(actionString.encode("UTF-8"))

        receivedData = sock.recv(1024).decode("UTF-8")
        print("recieved data", receivedData)

        while True:
            if input("continue?(y/n)").strip() == "y":
                break


main()
