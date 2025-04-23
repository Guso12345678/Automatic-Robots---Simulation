import numpy as np

class WallFollower:
    def __init__(self, dt: float):
        self._dt = dt
        self.Kp = 0.6
        self.Kd = 0.07
        self.Kw = 0.1
        self.setpoint = 0.5
        self.previous_error = 0.0
        self.state = "FOLLOW_WALL"
        self.turning_speed = 1.0
        self.turning_duration = np.pi / 2 / self.turning_speed  # Time for turning 90 degrees
        self.turn_timer = 0
        self.turn_direction = 0

    def _get_valid_min(self, scan_data):
        valid_data = [x for x in scan_data if np.isfinite(x) and x > 0]
        return min(valid_data) if valid_data else float("inf")

    def compute_commands(self, z_scan: list[float], z_v: float, z_w: float):
        front_distance = self._get_valid_min(z_scan[-10:] + z_scan[:10])
        left_distance = self._get_valid_min(z_scan[10:45])
        right_distance = self._get_valid_min(z_scan[195:230])

        if self.state == "FOLLOW_WALL":
            error = left_distance - right_distance
            derivative = (error - self.previous_error) / self._dt
            w = self.Kp * error + self.Kd * derivative - self.Kw * z_w
            self.previous_error = error
            v = max(z_v, 0.2)

            if front_distance < 0.2:
                self.state = "DETECT_TURN"
                self.turn_timer = 0
            return v, w

        elif self.state == "DETECT_TURN":
            if left_distance > right_distance:
                self.turn_direction = 1
            else:
                self.turn_direction = -1

            self.state = "TURNING"
            self.turn_timer = 0
            return 0.0, self.turn_direction * self.turning_speed

        elif self.state == "TURNING":
            if self.turn_timer < self.turning_duration:
                self.turn_timer += self._dt
                return 0.0, self.turn_direction * self.turning_speed
            else:
                self.state = "FOLLOW_WALL"
                self.turn_timer = 0
                return 0.0, 0.0

        return z_v, 0.0

