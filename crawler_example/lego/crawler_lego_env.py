#!/usr/bin/env pybricks-micropython

try:
    from pybricks import ev3brick as brick
    from pybricks.ev3devices import Motor, InfraredSensor
    from pybricks.parameters import Port, Stop, Button
    from pybricks.tools import print
    from pybricks.tools import wait
    import utils_motor
    print('Running on Robot')
    running_on_lego = True
except ModuleNotFoundError:
    print('Running on PC')
    running_on_lego = False


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

    @staticmethod
    def desc(val):
        if val == MotorState.NEUTRAL:
            return 'NEUTRAL'
        elif val == MotorState.UP:
            return 'UP'
        elif val == MotorState.DOWN:
            return 'DOWN'
        else:
            return 'INVALID STATE'


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

    @staticmethod
    def desc(val):
        if val == MotorType.LEG:
            return 'leg'
        elif val == MotorType.FEET:
            return 'feet'
        else:
            return 'INVALID STATE'


class CrawlingRobotEnv:
    """
    This class will implement a similar environment interface as seen on openAI gym, but actually
    interface with a real world-robot (Lego)
    """
    def __init__(self, invert_reward=False, step_angle=35, run_on_lego=True, no_change_penalty=-20):
        self.n_leg_state = 3
        self.n_feet_state = 3
        self.action_space = self.n_leg_state + self.n_feet_state
        self.observation_space = self.n_leg_state * self.n_feet_state
        # Set Initial state
        self.state = (MotorState.NEUTRAL, MotorState.NEUTRAL)
        self.running_on_lego = run_on_lego
        self.no_change_penalty = no_change_penalty
        if running_on_lego:
            # Initialize a motors
            self.leg_motor = Motor(Port.C)
            self.feet_motor = Motor(Port.A)
            # Initialize sensor for getting distance
            self.infrared = InfraredSensor(Port.S1)
        else:
            self.leg_motor = None
            self.feet_motor = None
            self.infrared = None

        self.motor_step_angle = step_angle
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

        # Dictionary that will retain samples of experience
        self.sampled_reward_function = {}
        self.sampled_mdp = {}

        # Populate dictionary to convert state tuple to state indexes
        self.state_2_index = self.__get_tuple_2_index()
        self.invert_reward = invert_reward

    def reset(self):
        """
        Reset robot internal states, and rotor angles, before calling this functions check
        if robot leg/feet are into neutral angles
        :return:
        """
        if self.running_on_lego:
            self.leg_motor.reset_angle(0.0)
            self.feet_motor.reset_angle(0.0)

        # Set Initial internal state
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
        if running_on_lego:
            # Get distance before move
            distance_before_move = self.infrared.distance()
        else:
            distance_before_move = 0

        # Send command to motors, and update state
        state = self.state
        no_state_change_penalty = self.__control_motors(motor_action)
        if running_on_lego:
            # Get distance travelled after motors did some job(reward)
            distance_after_move = self.infrared.distance()
            reward = distance_after_move - distance_before_move
        else:
            reward = self.mdp_immediate_reward(state, action)

        # Invert the reward if needed
        if self.invert_reward:
            reward *= -1

        # Penalty for doing command that doesn't change state
        reward += no_state_change_penalty

        next_state = self.state_2_index[self.state]

        # Save on dictionary the pair state, action, reward
        self.record_mdp(state, action, reward, next_state)

        # Return (next_state, reward, done, some_info)
        return next_state, reward, False, {}

    def __control_motors(self, motor_action):
        """
        Receive an action tupple (Motor, Action) and apply on the motor
        This function also need to take into account the safety of the robot, for example
        if the robot motor is already UP we will ignore another UP command to avoid breaking it
        :param motor_action: Input tupple
        :return: Next state after the action
        """
        no_change_penalty = 0
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
                if self.running_on_lego:
                    angle = self.dict_angle[curr_leg_pos, position]
                    utils_motor.leg(angle, self.leg_motor, self.feet_motor)
            else:
                no_change_penalty = self.no_change_penalty
        elif motor == MotorType.FEET:
            # Only change motor for different positions
            if position != curr_feet_pos:
                # Update state for motor 1
                self.state[1] = position
                if self.running_on_lego:
                    angle = self.dict_angle[curr_feet_pos, position]
                    utils_motor.feet(angle, self.feet_motor)
            else:
                no_change_penalty = self.no_change_penalty
        # Convert back to tuple
        self.state = tuple(self.state)
        return no_change_penalty

    def __get_tuple_2_index(self):
        """
        Utility function used to convert the internal state-space tuples ie:(NEUTRAL, UP) to state index
        0,1,2 ...
        :return: Index for state given one of possible state-space tuples.
        """
        leg_state_space, feet_state_space = self.n_leg_state, self.n_feet_state
        tuple_2_index = {}
        index = 0
        for x in range(leg_state_space):
            for y in range(feet_state_space):
                tuple_2_index[(MotorState.val(x), MotorState.val(y))] = index
                index += 1
        return tuple_2_index

    @staticmethod
    def mdp_immediate_reward(state, action):
        """
        Execute the immediate reward function (we get this from experiment actions manually on the robot)
        :param action:
        :param state:
        :return:
        Action table:
        0: (MotorType.LEG, MotorState.NEUTRAL),
        1: (MotorType.LEG, MotorState.UP),
        2: (MotorType.LEG, MotorState.DOWN),
        3: (MotorType.FEET, MotorState.NEUTRAL),
        4: (MotorType.FEET, MotorState.UP),
        5: (MotorType.FEET, MotorState.DOWN)

        State tuple: (LEG, FEET)
        """
        reward = 0
        if state == (MotorState.NEUTRAL, MotorState.NEUTRAL):
            if action == 5:
                reward = -10
            elif action == 2:
                reward = -1
        elif state == (MotorState.DOWN, MotorState.DOWN):
            if action == 4:
                reward = 10
            elif action == 1:
                reward = 5
        elif state == (MotorState.DOWN, MotorState.UP):
            if action == 5:
                reward = -10
        elif state == (MotorState.NEUTRAL, MotorState.DOWN):
            if action == 5:
                reward = -10
        elif state == (MotorState.NEUTRAL, MotorState.UP):
            if action == 5:
                reward = -10

        return reward

    def __str__(self):
        """
        Return the current internal state description
        :return:
        """
        arm_idx, feet_idx = self.state
        arm_desc = MotorState.desc(arm_idx)
        feet_desc = MotorState.desc(feet_idx)
        description_state = 'leg:' + arm_desc + ' feet:' + feet_desc
        return description_state

    def state_idx_to_str(self, state_idx):
        """
        Return the description of a state index
        :param state_idx: State index
        :return:
        """
        state = list(self.state_2_index.keys())[list(self.state_2_index.values()).index(state_idx)]
        arm_idx, feet_idx = state
        arm_desc = MotorState.desc(arm_idx)
        feet_desc = MotorState.desc(feet_idx)
        description_state = 'leg:' + arm_desc + ' feet:' + feet_desc
        return description_state

    def action_idx_to_str(self, action_idx):
        motor, motor_state = self.action_2_arm[action_idx]
        motor_str = MotorType.desc(motor)
        action_str = MotorState.desc(motor_state)
        description_action = motor_str + ' ' + action_str + ' idx:' + str(action_idx)
        return description_action

    def record_mdp(self, state, action, reward, next_state):
        """
        Use the previous state/action pairs
        :param state: Current state
        :param action: action
        :param reward: Immediate reward of doing an "action" at state "state"
        :param next_state: Next state after doing an "action" at state "state"
        :return: None
        """
        self.sampled_reward_function[state, action] = reward
        self.sampled_mdp[state, action] = next_state

    def read_sensor(self):
        return self.infrared.distance()
