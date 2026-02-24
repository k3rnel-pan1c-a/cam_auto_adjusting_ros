class CameraYOLOViewer(Node):
    def __init__(self):
        super().__init__('camera_yolo_viewer')
        self.bridge = CvBridge()

        pkg_share = get_package_share_directory('cam_auto_adjusting')
        pt_model_path = os.path.join(pkg_share, 'models', 'yolo26m_cam_one.pt')
        engine_path = os.path.join(pkg_share, 'models', 'yolo26m_cam_one.engine')

        if os.path.exists(engine_path):
            self.get_logger().info(f'Loading TensorRT engine: {engine_path}')
            self.model = YOLO(engine_path)
        else:
            self.get_logger().info('TensorRT engine not found, exporting from .pt model...')
            self.model = YOLO(pt_model_path)
            self.model.export(format='engine', device=0, half=True)
            self.model = YOLO(engine_path)

        cv2.namedWindow("YOLO", cv2.WINDOW_NORMAL)

        self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

    def image_callback(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        results = self.model.predict(frame, conf=0.1, verbose=False)

        annotated = results[0].plot()

        cv2.imshow("YOLO", annotated)
        cv2.waitKey(1)