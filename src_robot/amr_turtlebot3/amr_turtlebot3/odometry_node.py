import rclpy
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn
from rclpy.node import Node  # Importa CallbackReturn desde rclpy.node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from transforms3d.euler import quat2euler
import numpy as np

class OdometryNode(LifecycleNode):
    def __init__(self):
        super().__init__('odometry_node')
        self.declare_parameters(
            namespace='',
            parameters=[
                ('turning_speed', 1.0),
                ('kp', 4.5),
                ('kd', 2.5),
                ('kw', 0.1)
            ]
        )
        self.subscription = None
        self.publisher = None
        self.last_pose = None
        self.last_time = self.get_clock().now()

    def on_configure(self, state):
        self.get_logger().info("Configuring odometry_node...")
        try:
            self.subscription = self.create_subscription(
                Odometry,
                '/odom',
                self.odom_callback,
                10)
            self.get_logger().info("Subscription to /odom created successfully.")

            self.publisher = self.create_publisher(Odometry, '/odometry', 10)
            self.get_logger().info("Publisher to /odometry created successfully.")

            self.last_pose = None
            self.last_time = self.get_clock().now()
            self.get_logger().info("Node configured successfully.")
            return TransitionCallbackReturn.SUCCESS
        except Exception as e:
            self.get_logger().error('Failed to configure: {}'.format(str(e)))
            return TransitionCallbackReturn.ERROR


    def on_activate(self, state):
        if self.publisher:
            self.publisher.activate()
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state):
        if self.publisher:
            self.publisher.deactivate()
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state):
        if self.subscription:
            self.subscription.destroy()
        if self.publisher:
            self.publisher.destroy()
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state):
        if self.publisher:
            self.publisher.destroy()
        if self.subscription:
            self.subscription.destroy()
        return TransitionCallbackReturn.SUCCESS

    def odom_callback(self, msg):
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9

        if self.last_pose is not None and dt > 0:
            # Calcula el desplazamiento en x y y
            dx = msg.pose.pose.position.x - self.last_pose.pose.pose.position.x
            dy = msg.pose.pose.position.y - self.last_pose.pose.pose.position.y

            # Calcula la velocidad lineal
            linear_velocity = np.sqrt(dx**2 + dy**2) / dt

            # Calcula la rotación en el plano xy (yaw)
            _, _, last_yaw = quat2euler([
                self.last_pose.pose.pose.orientation.x,
                self.last_pose.pose.pose.orientation.y,
                self.last_pose.pose.pose.orientation.z,
                self.last_pose.pose.pose.orientation.w
            ])
            _, _, current_yaw = quat2euler([
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w
            ])

            # Calcula la velocidad angular
            angular_velocity = (current_yaw - last_yaw) / dt

            # Prepara el mensaje Odometry para ser publicado
            odometry_msg = Odometry()
            odometry_msg.header.stamp = current_time.to_msg()
            odometry_msg.header.frame_id = 'odom'
            odometry_msg.child_frame_id = 'base_link'

            # Establece la posición actual
            odometry_msg.pose.pose = msg.pose.pose

            # Establece las velocidades
            odometry_msg.twist.twist.linear.x = linear_velocity
            odometry_msg.twist.twist.angular.z = angular_velocity

            # Publica el mensaje de odometría
            self.publisher.publish(odometry_msg)

            # Guarda la última pose y el último tiempo para el próximo cálculo
            self.last_pose = msg
            self.last_time = current_time
        else:
            # Si es la primera vez, solo guarda la pose actual y el tiempo
            self.last_pose = msg
            self.last_time = current_time



def main(args=None):
    rclpy.init(args=args)
    node = OdometryNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
