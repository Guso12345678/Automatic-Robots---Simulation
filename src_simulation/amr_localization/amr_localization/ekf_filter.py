import numpy as np
from typing import Tuple


class EKFTracker:
    def __init__(self, dt: float, initial_pose: Tuple[float, float, float]):
        """Initialize the Extended Kalman Filter for pose tracking.

        Args:
            dt: Time step [s].
            initial_pose: Initial pose (x, y, theta) [m, m, rad].
        """
        # State: [x, y, theta]
        self._dt = dt
        self._state = np.array(initial_pose).reshape(3, 1)

        # Covariance matrix initialization
        self._P = np.diag([0.1, 0.1, 0.1])  # Initial uncertainty

        # Process noise - adjust these based on your robot's characteristics
        self._Q = np.diag([0.01, 0.01, 0.01])

        # Measurement noise - adjust based on sensor characteristics
        self._R = np.diag([0.1, 0.1, 0.1])

    def predict(self, v: float, w: float) -> None:
        """Prediction step of the EKF using the motion model.

        Args:
            v: Linear velocity [m/s].
            w: Angular velocity [rad/s].
        """
        x, y, theta = self._state.flatten()

        # Handle zero angular velocity case to avoid division by zero
        if abs(w) < 1e-6:
            # Straight line motion
            x_new = x + v * np.cos(theta) * self._dt
            y_new = y + v * np.sin(theta) * self._dt
            theta_new = theta
        else:
            # Circular motion
            x_new = x + (v / w) * (np.sin(theta + w * self._dt) - np.sin(theta))
            y_new = y + (v / w) * (np.cos(theta) - np.cos(theta + w * self._dt))
            theta_new = (theta + w * self._dt) % (2 * np.pi)

        # Check for valid values before updating state
        if not (np.isnan(x_new) or np.isnan(y_new) or np.isnan(theta_new)):
            # Update state
            self._state = np.array([x_new, y_new, theta_new]).reshape(3, 1)
        else:
            print("Warning: NaN detected in state prediction. Skipping update.")
            return

        # Compute Jacobian of motion model
        G = np.eye(3)
        if abs(w) < 1e-6:
            G[0, 2] = -v * np.sin(theta) * self._dt
            G[1, 2] = v * np.cos(theta) * self._dt
        else:
            G[0, 2] = (v / w) * (np.cos(theta + w * self._dt) - np.cos(theta))
            G[1, 2] = (v / w) * (np.sin(theta + w * self._dt) - np.sin(theta))

        # Update covariance
        self._P = G @ self._P @ G.T + self._Q

    def update(self, measurement: np.ndarray) -> None:
        """Update step of the EKF using sensor measurements.

        Args:
            measurement: Sensor measurement [x, y, theta].
        """
        # Check if measurement contains NaN values
        if np.any(np.isnan(measurement)):
            print("Warning: NaN detected in measurement. Skipping update.")
            return

        # Measurement matrix (direct observation of state)
        H = np.eye(3)

        # Kalman gain
        S = H @ self._P @ H.T + self._R

        # Check if S is invertible
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            print("Warning: Matrix S is not invertible. Using pseudo-inverse instead.")
            S_inv = np.linalg.pinv(S)

        K = self._P @ H.T @ S_inv

        # Update state
        innovation = measurement.reshape(3, 1) - self._state

        # Normalize angle difference to [-pi, pi]
        if not np.isnan(innovation[2]):
            innovation[2] = np.arctan2(np.sin(innovation[2]), np.cos(innovation[2]))
        else:
            innovation[2] = 0
            print("Warning: NaN detected in angle innovation. Setting to 0.")

        # Check for valid values before updating state
        if not np.any(np.isnan(K @ innovation)):
            self._state = self._state + K @ innovation
        else:
            print("Warning: NaN detected in state update calculation. Skipping update.")
            return

        # Update covariance
        self._P = (np.eye(3) - K @ H) @ self._P

    def get_pose(self) -> Tuple[float, float, float]:
        """Get the current estimated pose.

        Returns:
            Current pose estimate (x, y, theta) [m, m, rad].
        """
        return tuple(self._state.flatten())

    def get_covariance(self) -> np.ndarray:
        """Get the current state covariance matrix.

        Returns:
            3x3 covariance matrix.
        """
        return self._P
