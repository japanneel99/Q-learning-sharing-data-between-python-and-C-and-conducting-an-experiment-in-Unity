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

############ Running a  Deep q learning algorithm ####################################


class CustomEnvironment(Environment):
    def __init__(self, dt, pos_wheel_x, pos_wheel_y, pos_wheel_z, vel_wheel_x,
                 vel_wheel_y, pos_ped_x, pos_ped_y, pos_ped_z, vel_ped_x, vel_ped_y, vel_ped_z, feeling):
        super().__init__()

        self.dt = dt
        self.init_pos_wheel_x = pos_wheel_x
        self.init_pos_wheel_y = pos_wheel_y
        self.init_pos_wheel_z = pos_wheel_z
        self.init_vel_wheel_x = vel_wheel_x
        self.init_vel_wheel_y = vel_wheel_y
        self.init_pos_ped_x = pos_ped_x
        self.init_pos_ped_y = pos_ped_y
        self.init_pos_ped_z = pos_ped_z
        self.init_vel_ped_x = vel_ped_x
        self.init_vel_ped_y = vel_ped_y
        self.init_vel_ped_z = vel_ped_z
        self.init_feeling = feeling

    def states(self):
        return dict(type='float', shape=(11,))

    def actions(self):
        return dict(type='float', min_value=0, max_value=1.6)

    def close(self):
        super().close()

    def reset(self):
        self.state = np.array(
            [self.init_pos_wheel_x, self.init_pos_wheel_y, self.init_pos_wheel_z,
             self.init_vel_wheel_x, self.init_vel_wheel_y, self.init_pos_ped_x,
             self.init_pos_ped_y, self.init_pos_ped_z,
             self.init_vel_ped_x, self.init_vel_ped_y, self.init_vel_ped_z])

        return self.state

    def execute(self, actions):
        new_pos_wheel_x = self.state[3] * self.dt + self.state[0]
        new_pos_wheel_y = self.state[4] * self.dt + self.state[1]
        new_pos_wheel_z = actions * self.dt + self.state[2]
        new_pos_ped_x = self.state[8] * self.dt + self.state[5]
        new_pos_ped_y = self.state[9] * self.dt + self.state[6]
        new_pos_ped_z = self.state[10] * self.dt + self.state[7]

        terminal = (new_pos_wheel_z > new_pos_ped_z + 2)

        r = math.sqrt((new_pos_wheel_z - new_pos_ped_z) ** 2 +
                      (new_pos_wheel_x - new_pos_ped_x) ** 2)

        #rewards = self.init_feeling + actions - (1/(r))
        rewards = actions - (1/(r))
        self.state = np.array([new_pos_wheel_x, new_pos_wheel_y, new_pos_wheel_z,
                               self.init_vel_wheel_x, self.init_vel_wheel_y, new_pos_ped_x,
                               new_pos_ped_y, new_pos_ped_z, self.init_vel_ped_x,
                               self.init_vel_ped_y, self.init_vel_ped_z])

        return self.state, terminal, rewards


def main():
    host = "127.0.0.1"
    port = 25001

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))

    sock.sendall("GET_WHEELCHAIR_POSITION".encode("UTF-8"))
    receivedData = sock.recv(1024).decode("UTF-8")
    wheelchair_position_dict = json.loads(receivedData)
    wheelchair_position = [wheelchair_position_dict['x'],
                           wheelchair_position_dict['y'], wheelchair_position_dict['z']]

    sock.sendall("GET_WHEELCHAIR_VELOCITY".encode("UTF-8"))
    receivedData = sock.recv(1024).decode("UTF-8")
    wheelchair_vel_dict = json.loads(receivedData)
    wheelchair_velocity = [wheelchair_vel_dict['x'],
                           wheelchair_vel_dict['y'], wheelchair_vel_dict['z']]

    sock.sendall("GET_PEDESTRIAN_POSITION".encode("UTF-8"))
    receivedData = sock.recv(1024).decode("UTF-8")
    pedestrian_pos_dict = json.loads(receivedData)
    pedestrian_position = [pedestrian_pos_dict['x'],
                           pedestrian_pos_dict['y'], pedestrian_pos_dict['z']]

    sock.sendall("GET_PEDESTRIAN_VELOCITY".encode("UTF-8"))
    receivedData = sock.recv(1024).decode("UTF-8")
    pedestrian_vel_dict = json.loads(receivedData)
    pedestrian_velocity = [pedestrian_vel_dict['x'],
                           pedestrian_vel_dict['y'], pedestrian_vel_dict['z']]

    # ここには問題がある
    sock.sendall("Get_FEELING_REWARD".encode("UTF-8"))
    receivedData = sock.recv(1024).decode("UTF-8")
    feeling = []
    for i in receivedData:
        try:
            js = json.loads(i)
            feeling.append(js['info'])
        except Exception:
            pass

    print("Wheelchair_Position: ", wheelchair_position)
    print("Wheelchair_Velocity: ", wheelchair_velocity)
    print("Pedestrian_Position: ", pedestrian_position)
    print("Pedestrian_Velocity: ", pedestrian_velocity)
    print("feeling: ", feeling)

    pos_wheel_x = wheelchair_position[0]
    pos_wheel_y = wheelchair_position[1]
    pos_wheel_z = wheelchair_position[2]
    vel_wheel_x = wheelchair_velocity[0]
    vel_wheel_y = wheelchair_velocity[1]
    vel_wheel_z = wheelchair_velocity[2]
    pos_ped_x = pedestrian_position[0]
    pos_ped_y = pedestrian_position[1]
    pos_ped_z = pedestrian_position[2]
    vel_ped_x = pedestrian_velocity[0]
    vel_ped_y = pedestrian_velocity[1]
    vel_ped_z = pedestrian_velocity[2]

    dt = 0.15

    environment = CustomEnvironment(
        dt, pos_wheel_x, pos_wheel_y, pos_wheel_z, vel_wheel_x, vel_wheel_y,
        pos_ped_x, pos_ped_y, pos_ped_z, vel_ped_x, vel_ped_y, vel_ped_z, feeling)

    agent = Agent.create(
        agent='tensorforce', environment=environment, update=64,
        optimizer=dict(optimizer='adam', learning_rate=1e-3),
        objective='policy_gradient', reward_estimation=dict(horizon=20)
    )

    # print("test")
    n_epoch = 2000
    for i in range(n_epoch):
        states = environment.reset()
        terminal = False
        print("%d th learning....." % i)
        Wheelchair_position = []
        Pedestrian_position = []
        action_history = []
        cnt_step = 0

        while not terminal:
            cnt_step += 1
            actions = agent.act(states=states)
            states, terminal, reward = environment.execute(actions=actions)
            agent.observe(terminal=terminal, reward=reward)

            if i == n_epoch - 1:
                Wheelchair_position.append([states[0], states[1], states[2]])
                Pedestrian_position.append([states[5], states[6], states[7]])
                action_history.append(actions)
        print("number of step", cnt_step)

    Wheelchair_position = np.array(Wheelchair_position)
    Pedestrian_position = np.array(Pedestrian_position)

    print("Pedestrian_Position: ", Pedestrian_position)
    print("Wheelchair_Position: ", Wheelchair_position)
    print(reward)

    ########### plotting the position #################################
    plt.title("Wheelchair_position vs Pedestrian_Position")
    plt.xlabel("x_position")
    plt.ylabel("z_position")
    plt.scatter(
        Wheelchair_position[:, 0], Wheelchair_position[:, 2], label="Wheelchair_Position")
    plt.scatter(
        Pedestrian_position[:, 0], Pedestrian_position[:, 2], label="Pedestrian_Position")
    plt.legend()
    plt.show()

    ########### plotting action history(Velocities)###############
    plt.plot(action_history)
    plt.ylim([0, 2])
    plt.title("Wheelchair velocity, epoch = 2000 no feeling.")
    plt.xlabel("time_step")
    plt.ylabel("Wheelchair_velocity")
    plt.show()

    ########## Sending the data back to C# ######################
    actionString = "SEND_ACTION, 256"
    print(actionString)
    print("action history: ", action_history)

    ##################### plot an animation curve #######################################

    fig = plt.figure()

    def animate(i, Wheelchair_position, Pedestrian_position):
        print("drawing (%d/%d)" % (i, len(Wheelchair_position)))
        if i != 0:
            plt.cla()
        plt.xlim(0, 30)
        plt.ylim(-85, -50)
        plt.scatter(
            Wheelchair_position[i, 0], Wheelchair_position[i, 2], label="Wheelchair_Position")
        plt.scatter(
            Pedestrian_position[i, 0], Pedestrian_position[i, 2], label="Pedestrian_Position")

    animation = FuncAnimation(fig, animate, frames=len(
        Wheelchair_position), interval=100, fargs=(Wheelchair_position, Pedestrian_position))

    animation.save("dynamicPlot_epoch2000.png", writer="pillow")

    ###### Sending data to C# unity #################################
    while True:
        print("Sending Data", actionString)
        sock.sendall(actionString.encode("UTF-8"))

        receivedData = sock.recv(1024).decode("UTF-8")
        print("Received Data", receivedData)

        while True:
            if input("continue?(y/n)").strip() == "y":
                break


main()
