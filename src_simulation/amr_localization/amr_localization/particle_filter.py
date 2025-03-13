import datetime
import math
import numpy as np
import os
import pytz
import random

from amr_localization.maps import Map
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN


class ParticleFilter:
    """Particle filter implementation."""

    def __init__(
        self,
        dt: float,
        map_path: str,
        particle_count: int,
        sigma_v: float = 0.05,
        sigma_w: float = 0.1,
        sigma_z: float = 0.2,
        sensor_range_max: float = 8.0,
        sensor_range_min: float = 0.16,
        global_localization: bool = True,
        initial_pose: tuple[float, float, float] = (float("nan"), float("nan"), float("nan")),
        initial_pose_sigma: tuple[float, float, float] = (float("nan"), float("nan"), float("nan")),
    ):
        """Particle filter class initializer.

        Args:
            dt: Sampling period [s].
            map_path: Path to the map of the environment.
            particle_count: Initial number of particles.
            sigma_v: Standard deviation of the linear velocity [m/s].
            sigma_w: Standard deviation of the angular velocity [rad/s].
            sigma_z: Standard deviation of the measurements [m].
            sensor_range_max: Maximum sensor measurement range [m].
            sensor_range_min: Minimum sensor measurement range [m].
            global_localization: First localization if True, pose tracking otherwise.
            initial_pose: Approximate initial robot pose (x, y, theta) for tracking [m, m, rad].
            initial_pose_sigma: Standard deviation of the initial pose guess [m, m, rad].

        """
        self._dt: float = dt
        self._initial_particle_count: int = particle_count
        self._particle_count: int = particle_count
        self._sensor_range_max: float = sensor_range_max
        self._sensor_range_min: float = sensor_range_min
        self._sigma_v: float = sigma_v
        self._sigma_w: float = sigma_w
        self._sigma_z: float = sigma_z
        self._iteration: int = 0

        self._map = Map(
            map_path,
            sensor_range_max,
            compiled_intersect=True,
            use_regions=False,
            safety_distance=0.08,
        )
        self._particles = self._init_particles(
            particle_count, global_localization, initial_pose, initial_pose_sigma
        )
        self._figure, self._axes = plt.subplots(1, 1, figsize=(7, 7))
        self._timestamp = datetime.datetime.now(pytz.timezone("Europe/Madrid")).strftime(
            "%Y-%m-%d_%H-%M-%S"
        )

    def compute_pose(self) -> tuple[bool, tuple[float, float, float]]:
        """Computes the pose estimate when the particles form a single DBSCAN cluster.

        Adapts the amount of particles depending on the number of clusters during localization.
        100 particles are kept for pose tracking.

        Returns:
            localized: True if the pose estimate is valid.
            pose: Robot pose estimate (x, y, theta) [m, m, rad].

        """
        # TODO: 3.10. Complete the missing function body with your code.
        localized: bool = False
        pose: tuple[float, float, float] = (float("inf"), float("inf"), float("inf"))

        # Extract particle positions
        positions = np.array([[p[0], p[1]] for p in self._particles])

        # Perform DBSCAN clustering
        clustering = DBSCAN(eps=0.5, min_samples=5).fit(positions)
        labels = clustering.labels_

        # Count the number of clusters
        unique_labels = set(labels)
        num_clusters = len(unique_labels) - (1 if -1 in labels else 0)

        if num_clusters == 1:
            localized = True
            # Compute the centroid of the largest cluster
            largest_cluster = positions[labels == 0]
            centroid = largest_cluster.mean(axis=0)

            # Compute the average orientation of the particles in the largest cluster
            orientations = [
                self._particles[i][2] for i in range(len(self._particles)) if labels[i] == 0
            ]
            avg_orientation = np.arctan2(
                np.mean(np.sin(orientations)), np.mean(np.cos(orientations))
            )

            pose = (centroid[0], centroid[1], avg_orientation)
        else:
            # Reduce the number of particles if there are multiple clusters
            self._particle_count = max(100, self._particle_count // 2)
            self._particles = self._particles[: self._particle_count]

        return localized, pose

    def move(self, v: float, w: float) -> None:
        """Performs a motion update on the particles.

        Args:
            v: Linear velocity [m].
            w: Angular velocity [rad/s].

        """
        self._iteration += 1

        # TODO: 3.5. Complete the function body with your code.
        for i, particle in enumerate(self._particles):
            x, y, theta = particle

            v_noisy = v + np.random.normal(0, self._sigma_v)
            w_noisy = w + np.random.normal(0, self._sigma_w)

            x += v_noisy * self._dt * math.cos(theta)
            y += v_noisy * self._dt * math.sin(theta)
            theta += w_noisy * self._dt

            # Ensure theta is within [0, 2*pi)
            theta %= 2 * np.pi

            if not self._map.contains((x, y)):
                collision_result, _ = self._map.check_collision(
                    [(particle[0], particle[1]), (x, y)]
                )
                if collision_result:
                    x, y = collision_result

            self._particles[i] = (x, y, theta)

    def resample(self, measurements: list[float]) -> None:
        """Samples a new set of particles.

        Args:
            measurements: Sensor measurements [m].

        """
        # TODO: 3.9. Complete the function body with your code (i.e., replace the pass statement).
        weights = np.array(
            [self._measurement_probability(measurements, particle) for particle in self._particles]
        )
        weights += 1.0e-300
        weights /= weights.sum()

        indices = np.random.choice(
            range(self._particle_count), size=self._particle_count, p=weights
        )
        self._particles = self._particles[indices]

    def plot(self, axes, orientation: bool = True):
        """Draws particles.

        Args:
            axes: Figure axes.
            orientation: Draw particle orientation.

        Returns:
            axes: Modified axes.

        """
        if orientation:
            dx = [math.cos(particle[2]) for particle in self._particles]
            dy = [math.sin(particle[2]) for particle in self._particles]
            axes.quiver(
                self._particles[:, 0],
                self._particles[:, 1],
                dx,
                dy,
                color="b",
                scale=15,
                scale_units="inches",
            )
        else:
            axes.plot(self._particles[:, 0], self._particles[:, 1], "bo", markersize=1)

        return axes

    def show(
        self,
        title: str = "",
        orientation: bool = True,
        display: bool = False,
        block: bool = False,
        save_figure: bool = False,
        save_dir: str = "images",
    ):
        """Displays the current particle set on the map.

        Args:
            title: Plot title.
            orientation: Draw particle orientation.
            display: True to open a window to visualize the particle filter evolution in real-time.
                Time consuming. Does not work inside a container unless the screen is forwarded.
            block: True to stop program execution until the figure window is closed.
            save_figure: True to save figure to a .png file.
            save_dir: Image save directory.

        """
        figure = self._figure
        axes = self._axes
        axes.clear()

        axes = self._map.plot(axes)
        axes = self.plot(axes, orientation)

        axes.set_title(title + " (Iteration #" + str(self._iteration) + ")")
        figure.tight_layout()  # Reduce white margins

        if display:
            plt.show(block=block)
            plt.pause(0.001)  # Wait 1 ms or the figure won't be displayed

        if save_figure:
            save_path = os.path.realpath(
                os.path.join(os.path.dirname(__file__), "..", save_dir, self._timestamp)
            )

            if not os.path.isdir(save_path):
                os.makedirs(save_path)

            file_name = str(self._iteration).zfill(4) + " " + title.lower() + ".png"
            file_path = os.path.join(save_path, file_name)
            figure.savefig(file_path)

    def _init_particles(
        self,
        particle_count: int,
        global_localization: bool,
        initial_pose: tuple[float, float, float],
        initial_pose_sigma: tuple[float, float, float],
    ) -> np.ndarray:
        """Draws N random valid particles.

        The particles are guaranteed to be inside the map and
        can only have the following orientations [0, pi/2, pi, 3*pi/2].

        Args:
            particle_count: Number of particles.
            global_localization: First localization if True, pose tracking otherwise.
            initial_pose: Approximate initial robot pose (x, y, theta) for tracking [m, m, rad].
            initial_pose_sigma: Standard deviation of the initial pose guess [m, m, rad].

        Returns: A NumPy array of tuples (x, y, theta) [m, m, rad].

        """
        particles = np.empty((particle_count, 3), dtype=object)
        x_min, y_min, x_max, y_max = self._map.bounds()
        orientations = [0, np.pi / 2, np.pi, 3 * np.pi / 2]

        # TODO: 3.4. Complete the missing function body with your code.
        for i in range(particle_count):
            if global_localization:
                x, y = (
                    np.random.uniform(x_min, x_max),
                    np.random.uniform(y_min, y_max),
                )
                while not self._map.contains((x, y)):
                    x, y = (
                        np.random.uniform(x_min, x_max),
                        np.random.uniform(y_min, y_max),
                    )
            else:
                x = np.random.normal(initial_pose[0], initial_pose_sigma[0])
                y = np.random.normal(initial_pose[1], initial_pose_sigma[1])
                while not self._map.contains((x, y)):
                    x = np.random.normal(initial_pose[0], initial_pose_sigma[0])
                    y = np.random.normal(initial_pose[1], initial_pose_sigma[1])

            theta = np.random.choice(orientations)

            particles[i] = [x, y, theta]

        return particles

    def _sense(self, particle: tuple[float, float, float]) -> list[float]:
        """Obtains the predicted measurement of every LiDAR ray given the robot's pose.

        Args:
            particle: Particle pose (x, y, theta) [m, m, rad].

        Returns: List of predicted measurements; nan if a sensor is out of range.

        """
        z_hat: list[float] = []

        # TODO: 3.6. Complete the missing function body with your code.
        rays = self._lidar_rays(particle, range(0, 240, 30))  # 8 rays
        for ray in rays:
            intersection, distance = self._map.check_collision(ray, True)
            if intersection:
                z_hat.append(distance)
            else:
                z_hat.append(float("nan"))

        return z_hat

    @staticmethod
    def _gaussian(mu: float, sigma: float, x: float) -> float:
        """Computes the value of a Gaussian.

        Args:
            mu: Mean.
            sigma: Standard deviation.
            x: Variable.

        Returns:
            float: Gaussian value.

        """
        # TODO: 3.7. Complete the function body (i.e., replace the code below).
        exponent = -0.5 * ((x - mu) ** 2) / (sigma**2)
        return (1 / (sigma * math.sqrt(2 * math.pi))) * math.exp(exponent)

    def _lidar_rays(
        self, pose: tuple[float, float, float], indices: tuple[float], degree_increment: float = 1.5
    ) -> list[list[tuple[float, float]]]:
        """Determines the simulated LiDAR ray segments for a given robot pose.

        Args:
            pose: Robot pose (x, y, theta) in [m] and [rad].
            indices: Rays of interest in counterclockwise order (0 for to the forward-facing ray).
            degree_increment: Angle difference of the sensor between contiguous rays [degrees].

        Returns: Ray segments. Format:
                 [[(x0_start, y0_start), (x0_end, y0_end)],
                  [(x1_start, y1_start), (x1_end, y1_end)],
                  ...]

        """
        x, y, theta = pose

        # Convert the sensor origin to world coordinates
        x_start = x - 0.035 * math.cos(theta)
        y_start = y - 0.035 * math.sin(theta)

        rays = []

        for index in indices:
            ray_angle = math.radians(degree_increment * index)
            x_end = x_start + self._sensor_range_max * math.cos(theta + ray_angle)
            y_end = y_start + self._sensor_range_max * math.sin(theta + ray_angle)
            rays.append([(x_start, y_start), (x_end, y_end)])

        return rays

    def _measurement_probability(
        self, measurements: list[float], particle: tuple[float, float, float]
    ) -> float:
        """Computes the probability of a set of measurements given a particle's pose.

        If a measurement is unavailable (usually because it is out of range), it is replaced with
        the minimum sensor range to perform the computation because the environment is smaller
        than the maximum range.

        Args:
            measurements: Sensor measurements [m].
            particle: Particle pose (x, y, theta) [m, m, rad].

        Returns:
            float: Probability.

        """
        probability = 1.0

        # TODO: 3.8. Complete the missing function body with your code.
        predicted_measurements = self._sense(particle)
        for z, z_hat in zip(measurements[::30], predicted_measurements):
            if math.isnan(z):
                z = self._sensor_range_min
            if math.isnan(z_hat):
                z_hat = self._sensor_range_min
            probability = self._gaussian(z, self._sigma_z, z_hat)

        return probability
