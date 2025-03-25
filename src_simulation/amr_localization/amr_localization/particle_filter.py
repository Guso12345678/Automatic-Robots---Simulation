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
        localized: bool = False
        pose: tuple[float, float, float] = (float("inf"), float("inf"), float("inf"))
        
        if len(self._particles) == 0:
            return localized, pose

        particle_positions = np.array([[p[0], p[1]] for p in self._particles])
        valid_mask = ~np.isnan(particle_positions).any(axis=1)
        if not np.any(valid_mask):
            return localized, pose  

        particle_positions = particle_positions[valid_mask]
        valid_particles = self._particles[valid_mask]


        clustering = DBSCAN(eps=0.1, min_samples=10, algorithm='kd_tree', n_jobs = -1).fit(particle_positions)
        labels = clustering.labels_
        unique_labels = set(labels)
        unique_labels.discard(-1)

        if len(unique_labels) == 1:
            localized = True
            main_cluster = valid_particles[labels == list(unique_labels)[0]]
            x_mean = np.mean(main_cluster[:, 0])
            y_mean = np.mean(main_cluster[:, 1])
            sin_mean = np.mean(np.sin(main_cluster[:, 2]))
            cos_mean = np.mean(np.cos(main_cluster[:, 2]))
            theta_mean = np.arctan2(sin_mean, cos_mean)
            theta_mean %= 2 * np.pi
            pose = (x_mean, y_mean, theta_mean)
            self._particles = main_cluster[np.random.choice(len(main_cluster), 100, replace=True)]
        return localized, pose


    def move(self, v: float, w: float) -> None:
        self._iteration += 1
        for i, particle in enumerate(self._particles):
            x, y, theta = particle
            if any(np.isnan([x, y, theta])):
                continue

            v_noisy = v + np.random.normal(0, self._sigma_v)
            w_noisy = w + np.random.normal(0, self._sigma_w)
            v_noisy = np.clip(v_noisy, -1.0, 1.0)
            w_noisy = np.clip(w_noisy, -2.0, 2.0)
            x_new = x + v_noisy * self._dt * math.cos(theta)
            y_new = y + v_noisy * self._dt * math.sin(theta)
            theta_new = theta + w_noisy * self._dt
            theta_new %= 2 * np.pi

            if any(np.isnan([x_new, y_new, theta_new])):
                continue

            if not self._map.contains((x_new, y_new)):
                collision_result, _ = self._map.check_collision([(x, y), (x_new, y_new)])
                if collision_result:
                    x_new, y_new = collision_result
                else:
                    x_new, y_new = x, y

            self._particles[i] = (x_new, y_new, theta_new)

        self._particles = self._particles[~np.isnan(self._particles).any(axis=1)]

    def resample(self, measurements: list[float]) -> None:
        """Samples a new set of particles.

        Args:
            measurements: Sensor measurements [m].

        """
        # TODO: 3.9. Complete the function body with your code (i.e., replace the pass statement).
        weights = np.array(
            [self._measurement_probability(measurements, particle) for particle in self._particles]
        )
        weights /= np.sum(weights)
        N = len(self._particles)
        u1 = np.random.uniform(0, 1 / N)
        indexes = np.zeros(N, dtype=int)
        cumulative_sum = np.cumsum(weights)
        for k in range(1, N):
            u = u1 + (k - 1) / N
            indexes[k - 1] = np.searchsorted(cumulative_sum, u)

        self._particles = self._particles[indexes]

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
            """Draws N random valid particles more efficiently.

            The particles are guaranteed to be inside the map and
            can only have the following orientations [0, pi/2, pi, 3*pi/2].

            Args:
                particle_count: Number of particles.
                global_localization: First localization if True, pose tracking otherwise.
                initial_pose: Approximate initial robot pose (x, y, theta) for tracking [m, m, rad].
                initial_pose_sigma: Standard deviation of the initial pose guess [m, m, rad].

            Returns: A NumPy array of tuples (x, y, theta) [m, m, rad].
            """
            
            particles = np.empty((particle_count, 3), dtype=float)
            min_x, min_y, max_x, max_y = self._map.bounds()
            valid_orientations = [0, np.pi/2, np.pi, 3*np.pi/2]

            # TODO: 3.4. Complete the missing function body with your code.

            valid_particles = 0

            if global_localization:
                while valid_particles < particle_count:
                    
                    x = random.uniform(min_x, max_x)
                    y = random.uniform(min_y, max_y)
                    theta = random.choice(valid_orientations)
                    
                    if self._map.contains((x, y)):
                        particles[valid_particles] = (x, y, theta)
                        valid_particles += 1

            else:
                while valid_particles < particle_count:
                    x = random.gauss(initial_pose[0], initial_pose_sigma[0])
                    y = random.gauss(initial_pose[1], initial_pose_sigma[1])
                    theta = random.gauss(initial_pose[2], initial_pose_sigma[2])

                    if self._map.contains((x, y)):
                        particles[valid_particles] = (x, y, theta)
                        valid_particles += 1
            
            return particles

    def _sense(self, particle: tuple[float, float, float]) -> list[float]:
        """Obtains the predicted measurement of every LiDAR ray given the robot's pose.

        Args:
            particle: Particle pose (x, y, theta) [m, m, rad].

        Returns: List of predicted measurements; nan if a sensor is out of range.

        """
        z_hat: list[float] = []

        # TODO: 3.6. Complete the missing function body with your code.
        rango = range(0, 240, 30)
        rays = self._lidar_rays(particle, rango)

        for ray in rays:
            collision_point, distance = self._map.check_collision(ray, True)
            if collision_point:
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
        # TODO. 3.8. Complete the function body with your code.
        probability = 1.0
        predicted_measurements = self._sense(particle)
        indexed_measurements = measurements[::30]
        if len(indexed_measurements) != len(predicted_measurements):
            raise ValueError(
                "The length of indexed measurements and predicted measurements must be the same."
            )

        for z, z_hat in zip(indexed_measurements, predicted_measurements):
            if math.isnan(z):
                z = self._sensor_range_min
            if math.isnan(z_hat):
                z_hat = self._sensor_range_min

            prob = self._gaussian(z, self._sigma_z, z_hat)
            probability *= prob

        return probability
