#!/usr/bin/env pybricks-micropython

from pybricks import ev3brick as brick 
from pybricks.ev3devices import Motor, InfraredSensor
from pybricks.parameters import Port, Stop, Button
from pybricks.tools import print

import utils_motor

# Play a beep sound
brick.sound.beep()
print('Should display on VisualStudio')

# Clear the display
brick.display.clear()
brick.display.text("Hello", (0, 20))

# Initialize a motors and reset their angles
base_motor = Motor(Port.C)
leg_motor = Motor(Port.A)
base_motor.reset_angle(0.0)
leg_motor.reset_angle(0.0)
step_angle = 10

print('Hello Robot')

while True:
    # Get current distance
    #distance = InfraredSensor.distance()
    # Get current angles
    angle_base, angle_leg = utils_motor.get_curr_angle(base_motor, leg_motor)
    print('Angle base:', angle_base)
    print('Angle leg:', angle_leg)
    # Move arm/leg depending on the button pressed
    if Button.UP in brick.buttons():
        utils_motor.base_arm(step_angle, base_motor, leg_motor)
    elif Button.DOWN in brick.buttons():
        utils_motor.base_arm(-step_angle, base_motor, leg_motor)
    elif Button.LEFT in brick.buttons():
        utils_motor.leg_arm(step_angle, leg_motor)
    elif Button.RIGHT in brick.buttons():
        utils_motor.leg_arm(-step_angle, leg_motor)