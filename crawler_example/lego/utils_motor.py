#!/usr/bin/env pybricks-micropython
from pybricks.parameters import Stop


def get_curr_angle(motor_1, motor_2):
    angle_1 = motor_1.angle()
    angle_2 = motor_2.angle()
    return angle_1, angle_2


def leg(desired_angle, motor_1, motor_2):
    """
    The base is couple with the leg mechanically so we need to move both at the same time to keep
    the leg position unchanged
    """
    motor_1.run_angle(200, desired_angle, Stop.HOLD, False)
    motor_2.run_angle(200, desired_angle, Stop.HOLD, True)


def feet(desired_angle, motor_2):
    """
    Control only the leg which is not coupled with the base, so we can just move it
    """
    motor_2.run_angle(200, desired_angle, Stop.HOLD, True)
