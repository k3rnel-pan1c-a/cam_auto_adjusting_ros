#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import os
from ament_index_python.packages import get_package_share_directory


class CameraYOLOViewer(Node):
    def __init__(self):
        super().__init__('camera_yolo_viewer')
        self.bridge = CvBridge()

        # ── load your model (use absolute or __file__-based path) ──
        # e.g. MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'best1_with_corner.pt')
        pkg_share = get_package_share_directory('cam_auto_adjusting')
        self.model = YOLO(os.path.join(pkg_share, 'models', 'best1_with_corner.pt'))

        # subscribe to the raw camera topic
        self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        self.get_logger().info('Subscribed to /camera/image_raw, running YOLO…')

    def image_callback(self, msg: Image):
        # convert ROS Image → OpenCV BGR
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # run inference (returns a list of Results; we just take [0])
        results = self.model(frame)[0]

        # use YOLO’s built-in plot() to draw boxes + labels
        annotated = results.plot()

        # show the annotated frame
        cv2.imshow('YOLO Detections', annotated)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = CameraYOLOViewer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
