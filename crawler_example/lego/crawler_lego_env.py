#!/usr/bin/env pybricks-micropython

from pybricks import ev3brick as brick
from pybricks.ev3devices import Motor, InfraredSensor
from pybricks.parameters import Port, Stop, Button
from pybricks.tools import print
import utils_motor


class MotorState:
    """
    Possible positions this robot motor will have
    we're doing this because there is no enum on Micropython
    """
    NEUTRAL = 0
    UP = 1
    DOWN = 2

    @staticmethod
    def val(val):
        if val in [MotorState.NEUTRAL, MotorState.UP,  MotorState.DOWN]:
            return val
        else:
            return None


class MotorType:
    """
    Possible motor types on robot
    we're doing this because there is no enum on Micropython
    """
    LEG = 0
    FEET = 1

    @staticmethod
    def val(val):
        if val in [MotorType.LEG, MotorType.FEET]:
            return val
        else:
            return None


class CrawlingRobotEnv:
    """
    This class will implement a similar environment interface as seen on openAI gym, but actually
    interface with a real world-robot (Lego)
    """
    def __init__(self, invert_reward=False):
        self.n_leg_state = 3
        self.n_feet_state = 3
        self.action_space = self.n_leg_state + self.n_feet_state
        self.observation_space = self.n_leg_state * self.n_feet_state
        # Set Initial state
        self.state = (MotorState.NEUTRAL, MotorState.NEUTRAL)
        # Initialize a motors
        self.leg_motor = Motor(Port.C)
        self.feet_motor = Motor(Port.A)
        # Initialize sensor for getting distance
        self.infrared = InfraredSensor(Port.S1)
        self.motor_step_angle = 30
        # Dictionary to convert action indexes to motor commands
        self.action_2_arm = {0: (MotorType.LEG, MotorState.NEUTRAL),
                             1: (MotorType.LEG, MotorState.UP),
                             2: (MotorType.LEG, MotorState.DOWN),
                             3: (MotorType.FEET, MotorState.NEUTRAL),
                             4: (MotorType.FEET, MotorState.UP),
                             5: (MotorType.FEET, MotorState.DOWN)}

        # Dictionary to get the angles to move from one motor state to another, the key tuple is (previous, requested)
        self.dict_angle = {
            (MotorState.NEUTRAL, MotorState.NEUTRAL): 0,
            (MotorState.NEUTRAL, MotorState.UP): self.motor_step_angle,
            (MotorState.NEUTRAL, MotorState.DOWN): -self.motor_step_angle,
            (MotorState.UP, MotorState.UP): 0,
            (MotorState.UP, MotorState.NEUTRAL): -self.motor_step_angle,
            (MotorState.UP, MotorState.DOWN): -2*self.motor_step_angle,
            (MotorState.DOWN, MotorState.DOWN): self.motor_step_angle,
            (MotorState.DOWN, MotorState.NEUTRAL): self.motor_step_angle,
            (MotorState.DOWN, MotorState.UP): 2*self.motor_step_angle,
        }
        # Populate dictionary to convert state tuple to state indexes
        self.state_2_index = self.__get_tuple_2_index()
        self.invert_reward = invert_reward

    def reset(self):
        """
        Reset robot internal states, and rotor angles, before calling this functions check
        if robot leg/feet are into neutral angles
        :return:
        """
        self.leg_motor.reset_angle(0.0)
        self.feet_motor.reset_angle(0.0)
        self.state = (MotorState.NEUTRAL, MotorState.NEUTRAL)
        return self.state_2_index[self.state]

    def step(self, action):
        """
        Our RL Agent will interface with our robot through this function, the agent will send an action index
        and the environment(this class) will answer back a tuple (next_state, reward, end_game, other_info)
        :param action: Action index from (0..self.action_space-1)
        :return: next_state, reward, False, other_info
        """
        motor_action = self.action_2_arm[action]
        # Get distance before move
        distance_before_move = self.infrared.distance()
        # Send command to motors, and update state
        self.__control_motors(motor_action)
        # Get distance travelled after motors did some job(reward)
        distance_after_move = self.infrared.distance()
        reward = distance_after_move - distance_before_move

        # Invert the reward if needed
        if self.invert_reward:
            reward *= -1

        # Return (next_state, reward, done, some_info)
        return self.state_2_index[self.state], reward, False, {}

    def __position_2_angle(self, position):
        """
        Decode a position enum (UP,NEUTRAL,DOWN) to motor angles
        :param position: (UP,NEUTRAL,DOWN) values from MotorState enum
        :return:
        """
        if position == MotorState.UP:
            return self.motor_step_angle
        elif position == MotorState.NEUTRAL:
            return 0
        elif position == MotorState.DOWN:
            return -self.motor_step_angle

    def __control_motors(self, motor_action):
        """
        Receive an action tupple (Motor, Action) and apply on the motor
        This function also need to take into account the safety of the robot, for example
        if the robot motor is already UP we will ignore another UP command to avoid breaking it
        :param motor_action: Input tupple
        :return: Next state after the action
        """
        motor, position = motor_action
        # Convert state to list, make changes and bring back to tuple
        self.state = list(self.state)
        # Get current leg/feet position
        curr_leg_pos = self.state[0]
        curr_feet_pos = self.state[1]
        if motor == MotorType.LEG:
            # Only change motor for different positions
            if position != curr_leg_pos:
                # Update state for motor 0
                self.state[0] = position
                angle = self.dict_angle[curr_leg_pos, position]
                utils_motor.leg(angle, self.leg_motor, self.feet_motor)
        elif motor == MotorType.FEET:
            # Only change motor for different positions
            if position != curr_feet_pos:
                # Update state for motor 1
                self.state[1] = position
                angle = self.dict_angle[curr_feet_pos, position]
                utils_motor.feet(angle, self.feet_motor)
        # Convert back to tuple
        self.state = tuple(self.state)
        return self.state

    def __get_tuple_2_index(self):
        """
        Utility function used to convert the internal state-space tuples ie:(NEUTRAL, UP) to state index
        0,1,2 ...
        :return: Index for state given one of possible state-space tuples.
        """
        arm_state_space, leg_state_space = self.n_leg_state, self.n_feet_state
        tuple_2_index = {}
        index = 0
        for x in range(arm_state_space):
            for y in range(leg_state_space):
                tuple_2_index[(MotorState.val(x), MotorState.val(y))] = index
                index += 1
        return tuple_2_index
