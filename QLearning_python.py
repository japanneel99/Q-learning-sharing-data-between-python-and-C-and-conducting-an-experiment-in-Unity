import io
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
import math
import matplotlib.pyplot as plt
import csv


class QLearning:
    def __init__(self, dt):
        host = "127.0.0.1"
        port = 25001
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((host, port))

        self.dt = dt
        self.actions = np.array(
            [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6])
        self.n_quad_distance = 10
        self.n_quad_theta = 6
        self.n_vel = 2
        self.n_action = len(self.actions)
        self.max_distance = 10
        self.epsilon = 0.1

    def states(self):
        return dict(type='float', shape=(12,))

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
        # print(receivedData)

        return wheelchair_position, pedestrian_position, wheelchair_velocity, pedestrian_velocity, feeling

    def reset(self):
        print("resetting")
        self.sock.sendall("RESET".encode("UTF-8"))
        receivedData = self.sock.recv(1024).decode("UTF-8")

        wheelchair_position, pedestrian_position, wheelchair_velocity, pedestrian_velocity, feeling =\
            self.get_state_from_unity()
        self.state = np.array([wheelchair_position[0], wheelchair_position[1], wheelchair_position[2],
                               wheelchair_velocity[0], wheelchair_velocity[1], wheelchair_velocity[2],
                               pedestrian_position[0], pedestrian_position[1], pedestrian_position[2],
                               pedestrian_velocity[0], pedestrian_velocity[1], pedestrian_velocity[2]])

        print("Ended reset")
        return self.state

    def execute(self, actions):
        print("start executing")
        wheelchair_position, pedestrian_position, wheelchair_velocity, pedestrian_velocity, feeling =\
            self.get_state_from_unity()

        terminal = wheelchair_position[2] > pedestrian_position[2] + 2
        r = math.sqrt((wheelchair_position[2] - pedestrian_position[2]) ** 2 +
                      (wheelchair_position[0] - pedestrian_position[0]) ** 2)

        reward = feeling + actions - 2/r

        msg = "SEND_ACTION,%f" % actions
        self.sock.sendall(msg.encode("UTF-8"))
        receivedData = self.sock.recv(1024).decode("UTF-8")

        self.state = np.array([wheelchair_position[0], wheelchair_position[1], wheelchair_position[2],
                               wheelchair_velocity[0], wheelchair_velocity[1], wheelchair_velocity[2],
                               pedestrian_position[0], pedestrian_position[1], pedestrian_position[2],
                               pedestrian_velocity[0], pedestrian_velocity[1], pedestrian_velocity[2]])

        print("execute done")
        return self.state, reward, terminal

    def get_relative_distance(self, state):
        r_z = state[2] - state[8]
        r_x = state[6] - state[0]
        relative_distance = math.sqrt((r_z)**2 + (r_x)**2)
        return relative_distance

    def get_distance_index(self, state, max_distance, n_quad_distance):
        denominator = max_distance / n_quad_distance
        num = self.get_relative_distance(state)
        return max(0, min(self.n_quad_distance - 1, int(num/denominator)))

    def get_relative_angle(self, state):
        denominator = state[2] - state[8]
        numerator = state[6] - state[0]
        relative_angle = math.atan(numerator/denominator)
        return relative_angle

    def get_theta_index(self, state, n_quad_theta):
        denominator = 2*math.pi / n_quad_theta
        angle = self.get_relative_angle(state)
        numerator = angle + math.pi
        # return max(0, min(self.n_quad_theta - 3, int(numerator/denominator)))
        return int(numerator/denominator)

    def get_relative_xvelocity(self, state):
        rel_vx = state[3] - state[9]
        return rel_vx

    def get_velocity_index(self, state):
        vx_r = self.get_relative_xvelocity(state)
        if(vx_r > 0):
            return 1
        else:
            return 0

    def randomAction(self):
        return np.random.randint(self.n_action)

    def convert_state_to_index(self, state, n_quad_distance, n_quad_theta, max_distance):
        index_d = self.get_distance_index(state, max_distance, n_quad_distance)
        index_a = self.get_theta_index(
            state, n_quad_theta)
        index_v = self.get_velocity_index(state)

        index_distance = int(index_d)
        index_theta = int(index_a)
        index_vel = int(index_v)

        return index_distance, index_theta, index_vel


def main():
    calc_q_values = False
    dt = 0.01

    environment = QLearning(dt)

    actions = environment.actions
    n_quad_distance = environment.n_quad_distance
    n_quad_theta = environment.n_quad_theta
    n_vel = environment.n_vel
    max_distance = environment.max_distance
    n_action = environment.n_action
    epsilon = 0.1 if calc_q_values else 0.0
    gamma = 0.9
    alpha = 0.003

    if calc_q_values:
        # Q = np.zeros((n_quad_distance, n_quad_theta, n_vel, n_action))
        Q = np.load("a.npy")
    else:
        Q = np.load("b.npy")
        print("load")  # ok

    n_epoch = 40 if calc_q_values else 1
    for i in range(n_epoch):
        state = environment.reset()

        terminal = False
        print("%d th learning....." % i)
        Wheelchair_Position = []
        Pedestrian_Position = []
        action_history = []
        rewards = []

        cnt_step = 0

        while not terminal:
            cnt_step += 1
            index_distance, index_theta, index_vel = environment.convert_state_to_index(
                state, n_quad_distance, n_quad_theta, max_distance)

            if np.random.random() < epsilon:
                action_index = environment.randomAction()
            else:
                action_index = np.argmax(
                    Q[index_distance, index_theta, index_vel, :])

            action = environment.actions[action_index]
            next_state, reward, terminal = environment.execute(action)

            new_index_distance, new_index_theta, new_index_vel = environment.convert_state_to_index(
                next_state, n_quad_distance, n_quad_theta, max_distance)

            if calc_q_values:
                print(new_index_distance)
                print(new_index_theta)
                print(new_index_vel)
                print("length of Q:", len(Q))
                best_next_action_index = np.argmax(
                    Q[new_index_distance, new_index_theta, new_index_vel, :])
                td_target = reward + gamma * \
                    (Q[new_index_distance, new_index_theta,
                       new_index_vel, best_next_action_index])

                td_delta = td_target - Q[index_distance,
                                         index_theta, index_vel, action_index]

                Q[index_distance, index_theta, index_vel] = Q[index_distance,
                                                              index_theta, index_vel] + (alpha * td_delta)

            state = next_state

            if i == n_epoch - 1:
                Wheelchair_Position.append([state[0], state[1], state[2]])
                Pedestrian_Position.append([state[6], state[7], state[8]])
                action_history.append(action)
                rewards.append(reward)

    Wheelchair_Position = np.array(Wheelchair_Position)
    Pedestrian_Position = np.array(Pedestrian_Position)

    print("number of steps", cnt_step)
    print("Wheelchair_position: ", Wheelchair_Position)
    print("Pedestrian_Position: ", Pedestrian_Position)
    print("Action History", action_history)
    print("rewards: ", reward)

    print("Q values: ", Q)
    np.save("b.npy", Q)

    plt.plot
    plt.title("Wheelchair Position vs Pedestrian Position")
    plt.xlabel("x Position")
    plt.ylabel("z Position")
    plt.scatter(
        Wheelchair_Position[:, 0], Wheelchair_Position[:, 2], label="Wheelchair_Position")
    plt.scatter(
        Pedestrian_Position[:, 0], Pedestrian_Position[:, 2], label="Pedestrian_Position")
    plt.legend()
    plt.show()

    plt.title("Pedestrian and wheelchair x position")
    plt.plot(Wheelchair_Position[:, 0], label="Wheelchair x Position")
    plt.plot(Pedestrian_Position[:, 0], label="Pedestrian_x_position")
    plt.xlabel("time step")
    plt.ylabel("x position")
    plt.legend()
    plt.show()

    plt.plot(action_history)
    plt.ylim([0, 2])
    plt.title("Wheelchair Velocity")
    plt.xlabel("Time step")
    plt.ylabel = ("Wheelchair_Velocity")
    plt.show()

    plt.plot(rewards)
    plt.show()

    with open('csv_data_learned.csv', 'w', newline='') as csvfile:
        fieldnames = ['time_step', 'Wheelchair_Velocity']

        thewriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
        thewriter.writeheader()

        for i in range(len(action_history)):
            thewriter.writerow(
                {'time_step': i, 'Wheelchair_Velocity': action_history[i]})

    with open('csv_rewards_learned.csv', 'w', newline='') as csvfile:
        fieldnames = ['time_step', 'rewards']

        thewriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
        thewriter.writeheader()

        for i in range(len(rewards)):
            thewriter.writerow({'time_step': i, 'rewards': rewards[i]})


main()
