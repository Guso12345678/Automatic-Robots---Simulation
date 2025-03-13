# import numpy as np
# class WallFollower:
#     def __init__(self, dt: float):
#         self._dt = dt
#         self.Kp = 4.5 
#         self.Kd = 2.5
#         self.Kw = 0.1
#         self.setpoint = 0.2     # Distance from the wall
#         self.previous_error = 0.0   # PID control error
#         self.state = "FOLLOW_WALL"  # Initial state
#         self.turning_speed = 1.0
#         self.turning_duration = (np.pi)/ 4 / self.turning_speed  
#         self.turn_timer = 0
#         self.turn_direction = 0

#     def _get_valid_min(self, scan_data):
#         valid_data = [x for x in scan_data if np.isfinite(x) and x > 0]
#         return min(valid_data) if valid_data else float("inf")

#     def compute_commands(self, z_scan: list[float], z_v: float, z_w: float):
#         front_distance = self._get_valid_min(z_scan[-5:] + z_scan[:5])
#         left_distance = self._get_valid_min(z_scan[30:80])
#         right_distance = self._get_valid_min(z_scan[-30:-80])

#         if self.state == "FOLLOW_WALL":
#             error = left_distance - right_distance
#             derivative = (error - self.previous_error) / self._dt
#             w = self.Kp * error + self.Kd * derivative - self.Kw * z_w
#             self.previous_error = error
#             v = max(z_v, 0.2)

#             if front_distance < 0.2:
#                 self.state = "DETECT_TURN"
#                 self.turn_timer = 0
#             return v, w

#         elif self.state == "DETECT_TURN":
#             if left_distance > right_distance:
#                 self.turn_direction = 1
#             else:
#                 self.turn_direction = -1

#             self.state = "TURNING"
#             self.turn_timer = 0
#             return 0.0, self.turn_direction * self.turning_speed

#         elif self.state == "TURNING":
#             if self.turn_timer < self.turning_duration:
#                 self.turn_timer += self._dt
#                 return 0.0, self.turn_direction * self.turning_speed
#             else:
#                 self.state = "FOLLOW_WALL"
#                 self.turn_timer = 0
#                 return 0.0, 0.0

#         return z_v, 0.0

import math

class WallFollower:
    def __init__(self, dt: float) -> None:
        self._dt: float = dt
        self._desired_distance = 0.2
        self.Kp = 5
        self.Kd = 3.5
        self.Ki = 0.01
        self.integral_error = 0.0
        self.last_error = 0.0
        self._safety_distance = 0.25

        self._turn_left_mode = False
        self._turn_right_mode = False
        self._dead_end_mode = False
        self._rotation_completed = 0.0

        # Variables para almacenar el último valor válido
        self._last_front_distance = 0.15
        self._last_left_distance = 0.15
        self._last_right_distance = 0.15

    def compute_commands(self, z_scan: list[float], z_v: float, z_w: float) -> tuple[float, float]:
        front_distance = z_scan[0]
        left_distance = z_scan[60]
        right_distance = z_scan[-60]

        # Si es NaN, usa el último valor válido en lugar de asignar un valor fijo
        if math.isnan(front_distance):
            front_distance = self._last_front_distance
        else:
            self._last_front_distance = front_distance

        if math.isnan(left_distance):
            left_distance = self._last_left_distance
        else:
            self._last_left_distance = left_distance

        if math.isnan(right_distance):
            right_distance = self._last_right_distance
        else:
            self._last_right_distance = right_distance

        v = 0.1
        w = 0.0

        if (
            front_distance <= self._safety_distance
            and left_distance <= 0.2
            and right_distance <= 0.2
        ):
            self._dead_end_mode = False

        if front_distance <= self._safety_distance and not self._dead_end_mode:
            if right_distance >= left_distance:
                self._turn_right_mode = True
            else:
                self._turn_left_mode = True

        if self._turn_right_mode:
            print("voy a girar a la derecha")
            v = 0.0
            w = -1
            self._rotation_completed += abs(w) * self._dt
            if self._rotation_completed >= math.pi / 4:
                self._turn_right_mode = False
                self.last_error = 0
                self.integral_error = 0
                self._rotation_completed = 0.0
            return v, -w
        elif self._turn_left_mode:
            print("voy a girar a la izquierda")
            v = 0.0
            w = 1
            self._rotation_completed += abs(w) * self._dt
            if self._rotation_completed >= math.pi / 4:
                self._turn_left_mode = False
                self.last_error = 0
                self.integral_error = 0
                self._rotation_completed = 0.0
            return v, -w

        elif self._dead_end_mode:
            print("encerrado")
            v = 0.0
            w = 1
            self._rotation_completed += abs(w) * self._dt
            if self._rotation_completed >= math.pi / 2:
                self._dead_end_mode = False
                self.last_error = 0
                self.integral_error = 0
                self._rotation_completed = 0.0
            return v, w

        elif abs(left_distance - right_distance) < 0.2:
            if right_distance >= left_distance:
                error = left_distance - self._desired_distance
            else:
                error = self._desired_distance - right_distance
            derivative = (error - self.last_error) / self._dt
            self.integral_error += error * self._dt
            w = self.Kp * error + self.Kd * derivative + self.Ki * self.integral_error
            self.last_error = error
        return v, -w