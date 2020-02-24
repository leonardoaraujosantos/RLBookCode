"""
This project was developed by Peter Chen, Rocky Duan, Pieter Abbeel for the Berkeley Deep RL Bootcamp, August 2017.
Bootcamp website with slides and lecture videos: https://sites.google.com/view/deep-rl-bootcamp/.

Code adapted from CS188 project materials: http://ai.berkeley.edu/project_overview.html.

Copyright 2017 Deep RL Bootcamp Organizers.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of
the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
 INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

import numpy as np
from gym import Env, spaces
import math
from math import pi
import platform
import subprocess
import os
all_envs = []


class CrawlingRobotEnv(Env):

    def close_gui(self):
        if self.root is not None:
            self.root.destroy()
            self.root = None

    def __init__(self, horizon=np.inf, render=False, invert_reward=False, n_arms_state=7, n_hand_state=13, n_action=6):
        if render:
            import tkinter
            for env in all_envs:
                env.close_gui()

            all_envs.append(self)
            root = tkinter.Tk()
            root.title('Crawler GUI')
            root.resizable(0, 0)
            self.root = root
            canvas = tkinter.Canvas(root, height=200, width=1000)
            canvas.grid(row=2, columnspan=10)

            def close():
                if self.root is not None:
                    self.root.destroy()
                    self.root = None

            root.protocol('WM_DELETE_WINDOW', lambda: close)
            root.lift()
            root.attributes('-topmost', True)
            if platform.system() == 'Darwin':
                tmpl = 'tell application "System Events" to set frontmost of every process whose unix id is {} to true'
                script = tmpl.format(os.getpid())
                subprocess.check_call(['/usr/bin/osascript', '-e', script])
            root.attributes('-topmost', False)
        else:
            canvas = None
            self.root = None

        robot = CrawlingRobot(canvas)
        self.crawlingRobot = robot

        self._step_count = 0
        self.horizon = horizon

        # The state is of the form (armAngle, handAngle)
        # where the angles are bucket numbers, not actual
        # degree measurements
        self.state = None
        self.invert_reward = invert_reward

        # Number of possible states per actuator
        # On the Lego Robot it will be 3, 3
        self.nArmStates = n_arms_state
        self.nHandStates = n_hand_state
        self.tuple_2_state_idx = self.__get_tuple_2_index()

        # create a list of arm buckets and hand buckets to
        # discretize the state space
        min_arm_angle, max_arm_angle = self.crawlingRobot.get_min_max_arm_angles()
        min_hand_angle, max_hand_angle = self.crawlingRobot.get_min_max_hand_angles()
        arm_increment = (max_arm_angle - min_arm_angle) / (self.nArmStates - 1)
        hand_increment = (max_hand_angle - min_hand_angle) / (self.nHandStates - 1)
        self.armBuckets = [min_arm_angle + (arm_increment * i) for i in range(self.nArmStates)]
        self.handBuckets = [min_hand_angle + (hand_increment * i) for i in range(self.nHandStates)]

        self.action_space = spaces.Discrete(n_action)
        self.observation_space = spaces.Tuple(
            [spaces.Discrete(self.nArmStates), spaces.Discrete(self.nHandStates)]
        )

        # Reset
        self.reset()

    def __get_tuple_2_index(self):
        arm_state_space, leg_state_space = self.nArmStates, self.nHandStates
        tuple_2_index = {}
        index = 0
        for x in range(arm_state_space):
            for y in range(leg_state_space):
                tuple_2_index[(x, y)] = index
                index += 1
        return tuple_2_index

    @property
    def step_count(self):
        return self._step_count

    @step_count.setter
    def step_count(self, val):
        self._step_count = val
        self.crawlingRobot.draw(val, self.root)

    def _legal_actions(self, state):
        """
          Returns possible actions
          for the states in the
          current state
        """

        actions = list()

        curr_arm_bucket, curr_hand_bucket = state
        if curr_arm_bucket > 0:
            actions.append(0)
        if curr_arm_bucket < self.nArmStates - 1:
            actions.append(1)
        if curr_hand_bucket > 0:
            actions.append(2)
        if curr_hand_bucket < self.nHandStates - 1:
            actions.append(3)

        return actions

    def step(self, a):
        """
          Returns:
            s, r, d, info
        """
        if self.step_count >= self.horizon:
            raise Exception("Horizon reached")

        old_x, old_y = self.crawlingRobot.get_robot_position()
        arm_bucket, hand_bucket = self.state

        # TODO: Improvements for variable number of actions
        if a in self._legal_actions(self.state):
            if a == 0:
                new_arm_angle = self.armBuckets[arm_bucket - 1]
                self.crawlingRobot.move_arm(new_arm_angle)
                next_state = (arm_bucket - 1, hand_bucket)
            elif a == 1:
                new_arm_angle = self.armBuckets[arm_bucket + 1]
                self.crawlingRobot.move_arm(new_arm_angle)
                next_state = (arm_bucket + 1, hand_bucket)
            elif a == 2:
                new_hand_angle = self.handBuckets[hand_bucket - 1]
                self.crawlingRobot.move_hand(new_hand_angle)
                next_state = (arm_bucket, hand_bucket - 1)
            elif a == 3:
                new_hand_angle = self.handBuckets[hand_bucket + 1]
                self.crawlingRobot.move_hand(new_hand_angle)
                next_state = (arm_bucket, hand_bucket + 1)
            else:
                raise Exception("action out of range")
        else:
            next_state = self.state

        new_x, new_y = self.crawlingRobot.get_robot_position()

        # a simple reward function
        reward = new_x - old_x
        if self.invert_reward:
            # This will invert the direction
            reward *= -1

        self.state = next_state
        self.step_count += 1

        state_idx = self.tuple_2_state_idx[tuple(next_state)]

        return state_idx, reward, self.step_count >= self.horizon, {}

    def render(self, mode='human', close=False):
        pass

    def reset(self):
        """
         Resets the Environment to the initial state
        """
        arm_state = self.nArmStates // 2
        hand_state = self.nHandStates // 2
        self.state = arm_state, hand_state
        self.crawlingRobot.set_angles(self.armBuckets[arm_state], self.handBuckets[hand_state])
        self.crawlingRobot.positions = [20, self.crawlingRobot.get_robot_position()[0]]

        self.step_count = 0
        state_idx = self.tuple_2_state_idx[tuple(self.state)]
        return state_idx


class CrawlingRobot:
    """
    Implement drawing function and mechanics of simulated crawler robot
    """
    def __init__(self, canvas):

        # Canvas
        self.canvas = canvas
        self.velAvg = 0
        self.lastStep = 0

        # Arm and Hand Degrees
        self.arm_angle = self.oldArmDegree = 0.0
        self.hand_angle = self.oldHandDegree = -pi / 6

        self.maxArmAngle = pi / 6
        self.minArmAngle = -pi / 6

        self.maxHandAngle = 0
        self.minHandAngle = -(5.0 / 6.0) * pi

        self.robotWidth = 80
        self.robotHeight = 40
        self.armLength = 60
        self.handLength = 40
        self.positions = [0, 0]
        self.vel_avg_msg = None
        self.vel_msg = None
        self.pos_msg = None
        self.step_msg = None

        # Draw stuff
        if canvas is not None:
            self.totWidth = canvas.winfo_reqwidth()
            self.totHeight = canvas.winfo_reqheight()
            self.groundHeight = 40
            self.groundY = self.totHeight - self.groundHeight

            self.ground = canvas.create_rectangle(
                0,
                self.groundY, self.totWidth, self.totHeight, fill='blue'
            )

            # Robot Body
            self.robotPos = (self.totWidth / 5 * 2, self.groundY)
            self.robotBody = canvas.create_polygon(0, 0, 0, 0, 0, 0, 0, 0, fill='green')

            # Robot Arm
            self.robotArm = canvas.create_line(0, 0, 0, 0, fill='orange', width=5)

            # Robot Hand
            self.robotHand = canvas.create_line(0, 0, 0, 0, fill='red', width=3)

            # canvas.focus_force()

        else:
            self.robotPos = (20, 0)

    def set_angles(self, arm_angle, hand_angle):
        """
            set the robot's arm and hand angles
            to the passed in values
        """
        self.arm_angle = arm_angle
        self.hand_angle = hand_angle

    def get_angles(self):
        """
            returns the pair of (armAngle, handAngle)
        """
        return self.arm_angle, self.hand_angle

    def get_robot_position(self):
        """
            returns the (x,y) coordinates
            of the lower-left point of the
            robot
        """
        return self.robotPos

    def move_arm(self, new_arm_angle):
        """
            move the robot arm to 'newArmAngle'
        """
        if new_arm_angle > self.maxArmAngle:
            raise ValueError('Crawling Robot: Arm Raised too high. Careful!')
        if new_arm_angle < self.minArmAngle:
            raise ValueError('Crawling Robot: Arm Raised too low. Careful!')
        disp = self.displacement(self.arm_angle, self.hand_angle,
                                 new_arm_angle, self.hand_angle)
        cur_x_pos = self.robotPos[0]
        self.robotPos = (cur_x_pos + disp, self.robotPos[1])
        self.arm_angle = new_arm_angle

        # Position and Velocity Sign Post
        self.positions.append(self.get_robot_position()[0])
        if len(self.positions) > 100:
            self.positions.pop(0)

    def move_hand(self, new_hand_angle):
        """
            move the robot hand to 'newArmAngle'
        """

        if new_hand_angle > self.maxHandAngle:
            raise ValueError('Crawling Robot: Hand Raised too high. Careful!')
        if new_hand_angle < self.minHandAngle:
            raise ValueError('Crawling Robot: Hand Raised too low. Careful!')
        disp = self.displacement(self.arm_angle, self.hand_angle, self.arm_angle, new_hand_angle)
        cur_x_pos = self.robotPos[0]
        self.robotPos = (cur_x_pos + disp, self.robotPos[1])
        self.hand_angle = new_hand_angle

        # Position and Velocity Sign Post
        self.positions.append(self.get_robot_position()[0])
        if len(self.positions) > 100:
            self.positions.pop(0)

    def get_min_max_arm_angles(self):
        """
            get the lower- and upper- bound
            for the arm angles returns (min,max) pair
        """
        return self.minArmAngle, self.maxArmAngle

    def get_min_max_hand_angles(self):
        """
            get the lower- and upper- bound
            for the hand angles returns (min,max) pair
        """
        return self.minHandAngle, self.maxHandAngle

    def get_rotation_angle(self):
        """
            get the current angle the
            robot body is rotated off the ground
        """
        arm_cos, arm_sin = self.__get_cos_and_sin(self.arm_angle)
        hand_cos, hand_sin = self.__get_cos_and_sin(self.hand_angle)
        x = self.armLength * arm_cos + self.handLength * hand_cos + self.robotWidth
        y = self.armLength * arm_sin + self.handLength * hand_sin + self.robotHeight
        if y < 0:
            return math.atan(-y / x)
        return 0.0

    @staticmethod
    def __get_cos_and_sin(angle):
        return math.cos(angle), math.sin(angle)

    def displacement(self, old_arm_degree, old_hand_degree, arm_degree, hand_degree):

        old_arm_cos, old_arm_sin = self.__get_cos_and_sin(old_arm_degree)
        arm_cos, arm_sin = self.__get_cos_and_sin(arm_degree)
        old_hand_cos, old_hand_sin = self.__get_cos_and_sin(old_hand_degree)
        hand_cos, hand_sin = self.__get_cos_and_sin(hand_degree)

        x_old = self.armLength * old_arm_cos + self.handLength * old_hand_cos + self.robotWidth
        y_old = self.armLength * old_arm_sin + self.handLength * old_hand_sin + self.robotHeight

        x = self.armLength * arm_cos + self.handLength * hand_cos + self.robotWidth
        y = self.armLength * arm_sin + self.handLength * hand_sin + self.robotHeight

        if y < 0:
            if y_old <= 0:
                return math.sqrt(x_old * x_old + y_old * y_old) - math.sqrt(x * x + y * y)
            return (x_old - y_old * (x - x_old) / (y - y_old)) - math.sqrt(x * x + y * y)
        else:
            if y_old >= 0:
                return 0.0
            return -(x - y * (x_old - x) / (y_old - y)) + math.sqrt(x_old * x_old + y_old * y_old)

    def draw(self, step_count, root):
        if self.canvas is None or root is None:
            return
        x1, y1 = self.get_robot_position()
        x1 = x1 % self.totWidth

        # Check Lower Still on the ground
        if y1 != self.groundY:
            raise ValueError('Flying Robot!!')

        rotation_angle = self.get_rotation_angle()
        cos_rot, sin_rot = self.__get_cos_and_sin(rotation_angle)

        x2 = x1 + self.robotWidth * cos_rot
        y2 = y1 - self.robotWidth * sin_rot

        x3 = x1 - self.robotHeight * sin_rot
        y3 = y1 - self.robotHeight * cos_rot

        x4 = x3 + cos_rot * self.robotWidth
        y4 = y3 - sin_rot * self.robotWidth

        self.canvas.coords(self.robotBody, x1, y1, x2, y2, x4, y4, x3, y3)

        arm_cos, arm_sin = self.__get_cos_and_sin(rotation_angle + self.arm_angle)
        x_arm = x4 + self.armLength * arm_cos
        y_arm = y4 - self.armLength * arm_sin

        self.canvas.coords(self.robotArm, x4, y4, x_arm, y_arm)

        hand_cos, hand_sin = self.__get_cos_and_sin(self.hand_angle + rotation_angle)
        x_hand = x_arm + self.handLength * hand_cos
        y_hand = y_arm - self.handLength * hand_sin

        self.canvas.coords(self.robotHand, x_arm, y_arm, x_hand, y_hand)

        pos = self.positions[-1]
        velocity = pos - self.positions[-2]
        vel2 = (pos - self.positions[0]) / len(self.positions)
        self.velAvg = .9 * self.velAvg + .1 * vel2
        vel_msg = '100-step Avg Velocity: %.2f' % self.velAvg
        velocity_msg = 'Velocity: %.2f' % velocity
        position_msg = 'Position: %2.f' % pos
        step_msg = 'Step: %d' % step_count
        if 'vel_msg' in dir(self):
            self.canvas.delete(self.vel_msg)
            self.canvas.delete(self.pos_msg)
            self.canvas.delete(self.step_msg)
            self.canvas.delete(self.vel_avg_msg)

        self.vel_avg_msg = self.canvas.create_text(650, 190, text=vel_msg)
        self.vel_msg = self.canvas.create_text(450, 190, text=velocity_msg)
        self.pos_msg = self.canvas.create_text(250, 190, text=position_msg)
        self.step_msg = self.canvas.create_text(50, 190, text=step_msg)

        self.lastStep = step_count
        root.update()
