import numpy as np

class PurePursuit:
    """Class to follow a path using a simple pure pursuit controller."""

    def __init__(self, dt: float, lookahead_distance: float = 0.2):
        """Pure pursuit class initializer.

        Args:
            dt: Sampling period [s].
            lookahead_distance: Distance to the next target point [m].

        """
        self._dt: float = dt
        self._lookahead_distance: float = lookahead_distance
        self._path: list[tuple[float, float]] = []

    def compute_commands(self, x: float, y: float, theta: float) -> tuple[float, float]:
        """Pure pursuit controller implementation.

            Args:
            x: Estimated robot x coordinate [m].
            y: Estimated robot y coordinate [m].
            theta: Estimated robot heading [rad].

        Returns:
            v: Linear velocity [m/s].
            w: Angular velocity [rad/s].

        """
        # TODO: 4.11. Complete the body function with your code
        # Initialize velocities
        v = 0.0
        w = 0.0

        # Check if path exists
        if not self._path:
            return v, w

        # Find closest point on path and target point
        _, closest_idx = self._find_closest_point(x, y)
        target_xy = self._find_target_point((x, y), closest_idx)

        dx = target_xy[0] - x
        dy = target_xy[1] - y

        beta = np.arctan2(dy, dx)
        angle_diff = beta - theta
        alpha = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))

        rotation_threshold = np.pi / 8

        if abs(alpha) > rotation_threshold:
            v = 0.0
            w = alpha
            w = max(min(w, 1.0), -1.0)
        else:
            v = 0.1
            dist_l = np.hypot(dx, dy)

            if dist_l < 1e-3:
                w = 0.0
            else:
                w = 2 * v * np.sin(alpha) / dist_l
                w = max(min(w, 1.0), -1.0)

        return v, w

    @property
    def path(self) -> list[tuple[float, float]]:
        """Path getter."""
        return self._path

    @path.setter
    def path(self, value: list[tuple[float, float]]) -> None:
        """Path setter."""
        self._path = value

    def _find_closest_point(self, x: float, y: float) -> tuple[tuple[float, float], int]:
        """Find the closest path point to the current robot pose.

        Args:
            x: Estimated robot x coordinate [m].
            y: Estimated robot y coordinate [m].

        Returns:
            tuple[float, float]: (x, y) coordinates of the closest path point [m].
            int: Index of the path point found.

        """
        # TODO: 4.9. Complete the function body (i.e., find closest_xy and closest_idx).
        closest_xy = (0.0, 0.0)
        closest_idx = 0
        min_distance = np.inf

        for idx, (pos_x, pos_y) in enumerate(self._path):
            distance_squared = (x - pos_x) ** 2 + (y - pos_y) ** 2

            if distance_squared < min_distance:
                closest_idx = idx
                closest_xy = (pos_x, pos_y)
                min_distance = distance_squared

        return closest_xy, closest_idx

    def _find_target_point(
        self, origin_xy: tuple[float, float], origin_idx: int
    ) -> tuple[float, float]:
        """Find the destination path point based on the lookahead distance.

        Args:
            origin_xy: Current location of the robot (x, y) [m].
            origin_idx: Index of the current path point.

        Returns:
            tuple[float, float]: (x, y) coordinates of the target point [m].

        """
        # TODO: 4.10. Complete the function body with your code (i.e., determine target_xy).
        cumulative_dist = 0.0
        prev_x, prev_y = origin_xy

        for idx in range(origin_idx, len(self._path)):
            x, y = self._path[idx]
            segment_dist = np.hypot(x - prev_x, y - prev_y)
            cumulative_dist += segment_dist

            if cumulative_dist >= self._lookahead_distance:
                overshoot = cumulative_dist - self._lookahead_distance
                fraction = 1.0 - (overshoot / segment_dist)
                interp_x = prev_x + fraction * (x - prev_x)
                interp_y = prev_y + fraction * (y - prev_y)
                return (interp_x, interp_y)

            prev_x, prev_y = x, y
        return self._path[-1]
