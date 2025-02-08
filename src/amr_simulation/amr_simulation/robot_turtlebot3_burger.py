from amr_simulation.robot import Robot
from typing import Any
import numpy as np


class TurtleBot3Burger(Robot):
    """Class to control the Turtlebot3 Burger robot."""

    # Constants
    LINEAR_SPEED_MAX = 0.22  # Maximum linear velocity [m/s]
    SENSOR_RANGE_MAX = 8.0  # Maximum LiDAR sensor range [m]
    SENSOR_RANGE_MIN = 0.016  # Minimum LiDAR sensor range [m]
    TRACK = 0.16  # Distance between same axle wheels [m]
    WHEEL_RADIUS = 0.033  # Radius of the wheels [m]
    WHEEL_SPEED_MAX = LINEAR_SPEED_MAX / WHEEL_RADIUS  # Maximum motor angular speed [rad/s]

    def __init__(self, sim: Any, dt: float) -> None:
        """Turtlebot3 Burger robot class initializer.

        Args:
            sim: CoppeliaSim simulation handle.
            dt: Sampling period [s].

        """
        Robot.__init__(self, sim=sim, track=self.TRACK, wheel_radius=self.WHEEL_RADIUS)
        self._dt: float = dt
        self._motors: dict[str, int] = self._init_motors()

    def move(self, v: float, w: float) -> None:
        """Solve inverse differential kinematics and send commands to the motors.

        If the angular speed of any of the wheels is larger than the maximum admissible,
        sets the larger value to the maximum speed and proportionately scales the other.

        Args:
            v: Linear velocity of the robot center [m/s].
            w: Angular velocity of the robot center [rad/s].

        """
        array_velocidad = np.array([v, 0, w])

        # Crear la matriz de kinemática inversa
        matriz = np.array([
            [1/self.WHEEL_RADIUS, 0, (-self.TRACK * 0.5) / self.WHEEL_RADIUS],
            [1/self.WHEEL_RADIUS, 0, (self.TRACK * 0.5) / self.WHEEL_RADIUS]
        ])

        # Calcular las velocidades de las ruedas
        ruedas = np.dot(matriz, array_velocidad.transpose())
        velocidad_rueda_izq, velocidad_rueda_dcha = ruedas[0], ruedas[1]

        # Ajustar las velocidades si exceden el máximo permitido
        if abs(velocidad_rueda_izq) > self.WHEEL_SPEED_MAX:
            velocidad_rueda_izq = np.clip(velocidad_rueda_izq, -self.WHEEL_SPEED_MAX, self.WHEEL_SPEED_MAX)
            velocidad_rueda_dcha = (velocidad_rueda_dcha * velocidad_rueda_izq) / abs(velocidad_rueda_izq)

        if abs(velocidad_rueda_dcha) > self.WHEEL_SPEED_MAX:
            velocidad_rueda_dcha = np.clip(velocidad_rueda_dcha, -self.WHEEL_SPEED_MAX, self.WHEEL_SPEED_MAX)
            velocidad_rueda_izq = (velocidad_rueda_izq * velocidad_rueda_dcha) / abs(velocidad_rueda_dcha)

        # Enviar las velocidades a los motores
        self._sim.setJointTargetVelocity(self._motors["left"], velocidad_rueda_izq)
        self._sim.setJointTargetVelocity(self._motors["right"], velocidad_rueda_dcha)



    def sense(self) -> tuple[list[float], float, float]:
        """Read the LiDAR and the encoders.

        Returns:
            z_scan: Distance from every LiDAR ray to the closest obstacle in 1.5º increments [m].
            z_v: Linear velocity of the robot center [m/s].
            z_w: Angular velocity of the robot center [rad/s].

        """
        # Read LiDAR
        packed_data: str = self._sim.getBufferProperty(self._sim.handle_scene, "signal.lidar")
        z_scan: list[float] = self._sim.unpackFloatTable(packed_data)

        # Return nan if the measurement failed
        z_scan = [z if z >= 0.0 else float("nan") for z in z_scan]

        # Read encoders
        z_v, z_w = self._sense_encoders()

        return z_scan, z_v, z_w

    def _init_motors(self) -> dict[str, int]:
        """Acquire motor handles.

        Returns: {'left': handle, 'right': handle}

        """
        motors: dict[str, int] = {}

        motors["left"] = self._sim.getObject("/leftMotor")
        motors["right"] = self._sim.getObject("/rightMotor")

        return motors

    def _sense_encoders(self) -> tuple[float, float]:
        """Solve forward differential kinematics from encoder readings.

        Returns:
            z_v: Linear velocity of the robot center [m/s].
            z_w: Angular velocity of the robot center [rad/s].

        """
        # Read the angular position increment in the last sampling period [rad]
        encoders: dict[str, float] = {}

        encoders["left"] = self._sim.getFloatProperty(self._sim.handle_scene, "signal.leftEncoder")
        encoders["right"] = self._sim.getFloatProperty(
            self._sim.handle_scene, "signal.rightEncoder"
        )

        # TODO: 2.2. Compute the derivatives of the angular positions to obtain velocities [rad/s].
        velocity_angular_right = encoders["right"] / self._dt
        velocity_angular_left = encoders["left"] / self._dt


        # TODO: 2.3. Solve forward differential kinematics (i.e., calculate z_v and z_w).
        z_v = ((velocity_angular_left+velocity_angular_right)*self.WHEEL_RADIUS)/(2)
        z_w = ((velocity_angular_right-velocity_angular_left)*self.WHEEL_RADIUS)/(self.TRACK)
        
        return z_v, z_w
