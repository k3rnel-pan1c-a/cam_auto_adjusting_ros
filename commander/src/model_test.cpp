#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <rclcpp/rclcpp.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>

#include <yolos/yolos.hpp>
#include <opencv2/opencv.hpp>

#include <memory>

using Image = sensor_msgs::msg::Image;

using namespace std::placeholders;

class ModelTest {
public:
    ModelTest(const std::shared_ptr<rclcpp::Node>& node)
        : node_(node)
    {
        initYoloDetector();

        yolo_callback_group_ = node_->create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);
        
        auto sub_options = rclcpp::SubscriptionOptions();
        sub_options.callback_group = yolo_callback_group_;

        image_sub_ = node_->create_subscription<Image>(
            "camera/image_raw", 10,
            std::bind(&ModelTest::imageCallback, this, _1),
            sub_options);

        RCLCPP_INFO(node_->get_logger(), "ModelTest initialized for YOLO inference only");
    }

private:

    void initYoloDetector() {
        std::string model_path = node_->declare_parameter<std::string>(
            "yolo_model_path", ament_index_cpp::get_package_share_directory("cam_auto_adjusting") + "/models" + "/best1_with_corner.onnx");
        int num_classes = node_->declare_parameter<int>("yolo_num_classes", 7);
        bool use_gpu = node_->declare_parameter<bool>("yolo_use_gpu", false);
        
        conf_threshold_ = node_->declare_parameter<double>("yolo_conf_threshold", 0.05);
        iou_threshold_ = node_->declare_parameter<double>("yolo_iou_threshold", 0.05);

        std::string labels_path = "/tmp/yolo_labels.names";

        try {
            yolo_detector_ = std::make_unique<yolos::YOLODetector>(
                model_path, labels_path, use_gpu);
            
            class_names_ = yolo_detector_->getClassNames();

            RCLCPP_INFO(node_->get_logger(), 
                "YOLO detector initialized: model=%s, num_classes=%d, GPU=%s, conf_threshold: %.2f, iou_threshold: %.2f",
                model_path.c_str(), num_classes, use_gpu ? "true" : "false", conf_threshold_, iou_threshold_);
        } catch (const std::exception& e) {
            RCLCPP_ERROR(node_->get_logger(), 
                "Failed to initialize YOLO detector: %s", e.what());
        }
    }

    void imageCallback(const Image::SharedPtr msg) {
        if (!yolo_detector_) {
            return;
        }

        try {
            cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
            cv::Mat frame = cv_ptr->image;

            if (frame.empty()) {
                RCLCPP_WARN(node_->get_logger(), "Received empty image");
                return;
            }

            auto detections = yolo_detector_->detect(
                frame, 
                static_cast<float>(conf_threshold_), 
                static_cast<float>(iou_threshold_)
            );

            if (!detections.empty()) {
                RCLCPP_INFO(node_->get_logger(), 
                    "YOLO detected %zu objects", detections.size());
                
                yolo_detector_->drawDetections(frame, detections);
            }

            cv::imshow("detections", frame);
            cv::waitKey(1);

        } catch (const cv_bridge::Exception& e) {
            RCLCPP_ERROR(node_->get_logger(), 
                "cv_bridge exception: %s", e.what());
        } catch (const std::exception& e) {
            RCLCPP_ERROR(node_->get_logger(), 
                "YOLO inference exception: %s", e.what());
        }
    }

    std::shared_ptr<rclcpp::Node> node_;
    rclcpp::Subscription<Image>::SharedPtr image_sub_;

    rclcpp::CallbackGroup::SharedPtr yolo_callback_group_;

    std::unique_ptr<yolos::YOLODetector> yolo_detector_;
    double conf_threshold_{0.4};
    double iou_threshold_{0.45};

    std::vector<std::string> class_names_;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);

    auto node = std::make_shared<rclcpp::Node>("ModelTestNode");

    auto model_test = std::make_shared<ModelTest>(node);

    rclcpp::executors::MultiThreadedExecutor executor(
        rclcpp::ExecutorOptions(),
        2
    );
    executor.add_node(node);
    executor.spin();

    rclcpp::shutdown();
    return 0;
}
