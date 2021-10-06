import numpy as np
import time
import math
import csv
import random
import matplotlib.pyplot as plt


class PreTraining:
    def __init__(self, dt):

        self.dt = dt
        self.actions = np.array(
            [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6])
        self.n_quad_distance = 10
        self.n_quad_theta = 6
        self.n_vel = 2
        self.n_action = len(self.actions)
        self.max_distance = 7
        self.epsilon = 0.1

    def states(self):
        return dict(type='float', shape=(6,))

    def reset(self):
        self.pos_rx = 10.0+random.randint(-3, 3)
        self.pos_ry = 0.0 + random.randint(-3, 3)
        self.pos_rz = -10.0 + random.randint(-3, 3)
        self.vel_rx = 0.0
        self.vel_ry = 0.0
        self.vel_rz = 1.0
        self.pos_px = 1.0 + random.randint(-3, 3)
        self.pos_py = 0.0 + random.randint(-3, 3)
        self.pos_pz = 0.0 + random.randint(-3, 3)
        self.vel_px = 1.4
        self.vel_py = 0.0
        self.vel_pz = 0.0

        self.state = np.array(
            [self.pos_rx, self.pos_ry, self.pos_rz, self.pos_px, self.pos_py, self.pos_pz])
        return self.state

    def execute(self, actions):
        new_pos_rx = self.vel_rx * self.dt + self.pos_rx
        new_pos_ry = self.vel_ry * self.dt + self.pos_ry
        new_pos_rz = self.vel_rz * self.dt + self.pos_rz
        new_pos_px = self.vel_px * self.dt + self.pos_px
        new_pos_py = self.vel_py * self.dt + self.pos_py
        new_pos_pz = self.pos_pz * self.dt + self.pos_pz

        r = math.sqrt((new_pos_rz - new_pos_pz)**2 +
                      (new_pos_rx - new_pos_px)**2)

        terminal = new_pos_rz > 2

        reward = actions - (2/r)

        self.pos_rx = new_pos_rx
        self.pos_ry = new_pos_ry
        self.pos_rz = new_pos_rz
        self.pos_px = new_pos_px
        self.pos_py = new_pos_py
        self.pos_pz = new_pos_pz

        next_state = np.array(
            [self.pos_rx, self.pos_ry, self.pos_rz, self.pos_px, self.pos_py, self.pos_pz])
        return next_state, terminal, reward

    def get_relative_distance(self, rx, rz, px, pz):
        a = math.sqrt((rz - pz)**2 + (rx - px)**2)
        return a

    def get_distance_index(self, rx, rz, px, pz, max_distance, n_quad_distance):
        denominator = max_distance / n_quad_distance
        num = self.get_relative_distance(rx, rz, px, pz)
        return max(0, min(self.n_quad_distance-1, int(num/denominator)))

    def get_relative_angle(seld, rx, rz, px, pz):
        den = rz - pz
        num = rx - px
        return math.atan(num/den)

    def get_index_angle(self, rx, rz, px, pz, n_quad_theta):
        denominator = 2*math.pi / n_quad_theta
        angle = self.get_relative_angle(rx, rz, px, pz)
        num = angle + math.pi
        return int(num/denominator)

    def get_relative_xvelocity(self, vel_rx, vel_px):
        rel_vx = vel_rx - vel_px
        return rel_vx

    def get_velocity_index(self, vel_rx, vel_px):
        r_vx = vel_rx - vel_px
        if(r_vx > 0):
            return 1
        else:
            return 0

    def randomAction(self):
        return np.random.randint(self.n_action)

    def convert_state_to_index(self, state, n_quad_distance, n_quad_theta, max_distance, vel_rx, vel_px):

        self.vel_rx = vel_rx
        self.vel_px = vel_px
        index_d = self.get_distance_index(
            state[0], state[2], state[3], state[5], max_distance, n_quad_distance)
        index_a = self.get_index_angle(
            state[0], state[2], state[3], state[5], n_quad_theta)
        index_v = self.get_velocity_index(vel_rx=0.0, vel_px=1.0)

        index_distance = int(index_d)
        index_theta = int(index_a)
        index_vel = int(index_v)

        return index_distance, index_theta, index_vel


def main():
    calc_q_values = False
    dt = 0.01

    environment = PreTraining(dt)

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
        Q = np.zeros((n_quad_distance, n_quad_theta, n_vel, n_action))
    else:
        Q = np.load("a.npy")
        print("load")

    n_epoch = 750 if calc_q_values else 1
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
            # print("step: ", cnt_step)
            index_distance, index_theta, index_vel = environment.convert_state_to_index(
                state, n_quad_distance, n_quad_theta, max_distance, vel_rx=0.0, vel_px=1.0)

            if np.random.random() < epsilon:
                action_index = environment.randomAction()
            else:
                action_index = np.argmax(
                    Q[index_distance, index_theta, index_vel, :])

            action = environment.actions[action_index]
            next_state, terminal, reward = environment.execute(action)

            new_index_distance, new_index_theta, new_index_vel = environment.convert_state_to_index(
                next_state, n_quad_distance, n_quad_theta, max_distance, vel_rx=0.0, vel_px=1.0)

            if calc_q_values:
                best_next_action_index = np.argmax(
                    Q[new_index_distance, new_index_theta, new_index_vel, :])
                td_target = reward + gamma * (Q[new_index_distance, new_index_theta,
                                                new_index_vel, best_next_action_index])
                td_delta = td_target - Q[index_distance,
                                         index_theta, index_vel, action_index]

                Q[index_distance, index_theta, index_vel, action_index] = Q[index_distance,
                                                                            index_theta, index_vel, action_index] + (alpha * td_delta)

            state = next_state

            if i == n_epoch - 1:
                Wheelchair_Position.append([state[0], state[1]])
                Pedestrian_Position.append([state[2], state[3]])
                action_history.append(action)
                rewards.append(reward)

    Wheelchair_Position = np.array(Wheelchair_Position)
    Pedestrian_Position = np.array(Pedestrian_Position)

    print("Pedestrian_Position: ", Pedestrian_Position)
    print("Wheelchair_Position: ", Wheelchair_Position)
    print("Action History: ", action_history)
    print("Rewards: ", rewards)

    print("Q_values: ", Q)
    np.save("a.npy", Q)

    plt.plot(action_history)
    plt.show()

    plt.plot(rewards)
    plt.show()


main()
