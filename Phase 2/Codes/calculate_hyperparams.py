import math

def compute_rotation_parameters(omega_rad_per_s, direction='CCW'):
    # Constants
    wheel_circumference_cm = 17.6
    r = wheel_circumference_cm / (2 * math.pi)  # wheel radius in cm
    b = 4.8  # half the distance between wheels in cm

    # Ensure valid direction
    direction = direction.upper()
    if direction not in ['CW', 'CCW']:
        raise ValueError("Direction must be 'CW' or 'CCW'")

    # Set wheel speeds based on direction
    if direction == 'CCW':
        v_r = r * omega_rad_per_s
        v_l = -r * omega_rad_per_s
    else:  # CW
        v_r = -r * omega_rad_per_s
        v_l = r * omega_rad_per_s

    # Angular velocity of the robot
    u_omega = (r / b) * omega_rad_per_s

    # Time to rotate 90 degrees (π/2 radians)
    delta_t = (math.pi / 2) / u_omega

    return delta_t, v_l, v_r