#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.hpp>
#include <my_robot_interfaces/msg/pose_command.hpp>
#include <my_robot_interfaces/srv/get_current_pose.hpp>
#include <std_msgs/msg/string.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>

// YOLOs-CPP inference engine
#include <yolos/yolos.hpp>

#include <opencv2/opencv.hpp>

#include <cmath>
#include <atomic>
#include <memory>
#include <mutex>
#include <fstream>

using MoveGroupInterface = moveit::planning_interface::MoveGroupInterface;

using PoseCommand = my_robot_interfaces::msg::PoseCommand;
using GetCurrentPose = my_robot_interfaces::srv::GetCurrentPose;

using String = std_msgs::msg::String;
using LaserScan = sensor_msgs::msg::LaserScan;
using Image = sensor_msgs::msg::Image;

using namespace std::placeholders;


class Commander {
public:
    Commander(const std::shared_ptr<rclcpp::Node>& main_node,
              const std::shared_ptr<rclcpp::Node>& moveit_node)
        : main_node_(main_node),
          moveit_node_(moveit_node)

    {
        arm_group_ = std::make_shared<MoveGroupInterface>(
            moveit_node_,
            MoveGroupInterface::Options("arm_group", "robot_description", "")
        );

        arm_group_->setMaxAccelerationScalingFactor(1.0);
        arm_group_->setMaxVelocityScalingFactor(1.0);

        go_to_named_target_sub_ = main_node_->create_subscription<String>(
            "named_target", 10,
            std::bind(&Commander::namedTargetCmdCallback, this, _1));

        get_current_pose_service_ = main_node_->create_service<GetCurrentPose>(
            "get_current_pose",
            std::bind(&Commander::getCurrentPoseService, this, _1, _2));

        initYoloDetector();

        image_sub_ = main_node_->create_subscription<Image>(
            "camera/image_raw", 10,
            std::bind(&Commander::imageCallback, this, _1));

        RCLCPP_INFO(main_node_->get_logger(), "Commander initialized with YOLO detector");
    }

    bool goToNamedTarget(const std::string &name) {
        arm_group_->setStartStateToCurrentState();
        arm_group_->setNamedTarget(name);
        return planAndExecute(arm_group_);
    }

    bool goToPoseTarget(double x, double y, double z,
                        double roll, double pitch, double yaw,
                        bool cartesian_path = false)
    {
        tf2::Quaternion q;
        q.setRPY(roll, pitch, yaw);
        q.normalize();

        geometry_msgs::msg::PoseStamped target_pose;
        target_pose.header.frame_id = "base_link";
        target_pose.pose.position.x = x;
        target_pose.pose.position.y = y;
        target_pose.pose.position.z = z;
        target_pose.pose.orientation.x = q.getX();
        target_pose.pose.orientation.y = q.getY();
        target_pose.pose.orientation.z = q.getZ();
        target_pose.pose.orientation.w = q.getW();

        arm_group_->setStartStateToCurrentState();

        if (!cartesian_path) {
            arm_group_->setPoseTarget(target_pose);
            return planAndExecute(arm_group_);
        } else {
            std::vector<geometry_msgs::msg::Pose> waypoints;
            waypoints.push_back(target_pose.pose);

            moveit_msgs::msg::RobotTrajectory trajectory;
            double fraction = arm_group_->computeCartesianPath(
                waypoints, 0.01, trajectory);

            if (fraction == 1.0) {
                auto result = arm_group_->execute(trajectory);
                return (result == moveit::core::MoveItErrorCode::SUCCESS);
            }
            return false;
        }
    }

    bool goToPositionTarget(double x, double y, double z,
                            bool cartesian_path = false)
    {
        auto current_pose = arm_group_->getCurrentPose();

        tf2::Quaternion q(
            current_pose.pose.orientation.x,
            current_pose.pose.orientation.y,
            current_pose.pose.orientation.z,
            current_pose.pose.orientation.w
        );

        double roll, pitch, yaw;
        tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);

        return goToPoseTarget(x, y, z, roll, pitch, yaw, cartesian_path);
    }


    std::vector<yolos::Detection> getLatestDetections() {
        std::lock_guard<std::mutex> lock(detections_mutex_);
        return latest_detections_;
    }

    std::string getClassName(int classId) const {
        if (yolo_detector_ && classId >= 0 && 
            static_cast<size_t>(classId) < yolo_detector_->getClassNames().size()) {
            return yolo_detector_->getClassNames()[classId];
        }
        return "unknown";
    }

private:

    void initYoloDetector() {
        std::string model_path = main_node_->declare_parameter<std::string>(
            "yolo_model_path", "/home/cam_auto_adjusting_ws/src/cam_auto_adjusting/models/best1_with_corner.onnx");
        int num_classes = main_node_->declare_parameter<int>("yolo_num_classes", 7);
        bool use_gpu = main_node_->declare_parameter<bool>("yolo_use_gpu", false);
        
        conf_threshold_ = main_node_->declare_parameter<double>("yolo_conf_threshold", 0.4);
        iou_threshold_ = main_node_->declare_parameter<double>("yolo_iou_threshold", 0.45);

        std::string labels_path = "/tmp/yolo_labels.names";

        try {
            yolo_detector_ = std::make_unique<yolos::YOLODetector>(
                model_path, labels_path, use_gpu);
            
            RCLCPP_INFO(main_node_->get_logger(), 
                "YOLO detector initialized: model=%s, num_classes=%d, GPU=%s",
                model_path.c_str(), num_classes, use_gpu ? "true" : "false");
        } catch (const std::exception& e) {
            RCLCPP_ERROR(main_node_->get_logger(), 
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
                RCLCPP_WARN(main_node_->get_logger(), "Received empty image");
                return;
            }

            auto detections = yolo_detector_->detect(
                frame, 
                static_cast<float>(conf_threshold_), 
                static_cast<float>(iou_threshold_)
            );

            {
                std::lock_guard<std::mutex> lock(detections_mutex_);
                latest_detections_ = detections;
            }

            if (!detections.empty()) {
                RCLCPP_INFO(main_node_->get_logger(), 
                    "YOLO detected %zu objects", detections.size());
                
                for (const auto& det : detections) {
                    std::string class_name = getClassName(det.classId);
                    RCLCPP_DEBUG(main_node_->get_logger(),
                        "  - %s (%.2f%%) at [%d, %d, %d, %d]",
                        class_name.c_str(),
                        det.conf * 100.0f,
                        det.box.x, det.box.y, 
                        det.box.width, det.box.height);
                }
            }

        } catch (const cv_bridge::Exception& e) {
            RCLCPP_ERROR(main_node_->get_logger(), 
                "cv_bridge exception: %s", e.what());
        } catch (const std::exception& e) {
            RCLCPP_ERROR(main_node_->get_logger(), 
                "YOLO inference exception: %s", e.what());
        }
    }

    bool planAndExecute(const std::shared_ptr<MoveGroupInterface>& interface) {
        MoveGroupInterface::Plan plan;
        bool success = (interface->plan(plan) == moveit::core::MoveItErrorCode::SUCCESS);

        if (success)
            interface->execute(plan);
        
        return success;
    }

    void namedTargetCmdCallback(const String &msg) {
        goToNamedTarget(msg.data);
    }

    void getCurrentPoseService(
        const std::shared_ptr<GetCurrentPose::Request> request,
        std::shared_ptr<GetCurrentPose::Response> response)
    {
        (void)request;
        
        try {
            auto current_pose = arm_group_->getCurrentPose();
            
            response->x = current_pose.pose.position.x;
            response->y = current_pose.pose.position.y;
            response->z = current_pose.pose.position.z;
            
            tf2::Quaternion q(
                current_pose.pose.orientation.x,
                current_pose.pose.orientation.y,
                current_pose.pose.orientation.z,
                current_pose.pose.orientation.w
            );
            
            double roll, pitch, yaw;
            tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);
            
            response->roll = roll;
            response->pitch = pitch;
            response->yaw = yaw;
            
            response->success = true;
            response->message = "Successfully retrieved current pose";
            
            RCLCPP_INFO(main_node_->get_logger(),
                        "Service called - Current pose: position [%.3f, %.3f, %.3f] orientation [roll: %.3f, pitch: %.3f, yaw: %.3f]",
                        response->x, response->y, response->z,
                        response->roll, response->pitch, response->yaw);
        } catch (const std::exception& e) {
            response->success = false;
            response->message = std::string("Failed to get current pose: ") + e.what();
            RCLCPP_ERROR(main_node_->get_logger(), "%s", response->message.c_str());
        }
    }
    std::shared_ptr<rclcpp::Node> main_node_;
    std::shared_ptr<rclcpp::Node> moveit_node_;

    std::shared_ptr<MoveGroupInterface> arm_group_;

    rclcpp::Subscription<String>::SharedPtr go_to_named_target_sub_;
    rclcpp::Subscription<Image>::SharedPtr image_sub_;

    rclcpp::Service<GetCurrentPose>::SharedPtr get_current_pose_service_;

    // YOLO detector
    std::unique_ptr<yolos::YOLODetector> yolo_detector_;
    double conf_threshold_{0.4};
    double iou_threshold_{0.45};

    // Latest detections storage (thread-safe access)
    std::vector<yolos::Detection> latest_detections_;
    std::mutex detections_mutex_;
};


int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);

    auto main_node = std::make_shared<rclcpp::Node>("CommanderNode");
    auto moveit_node = std::make_shared<rclcpp::Node>("MoveItNode");


    auto commander = std::make_shared<Commander>(main_node, moveit_node);

    rclcpp::executors::MultiThreadedExecutor executor(
        rclcpp::ExecutorOptions(),
        4 
    );
    executor.add_node(main_node);
    executor.add_node(moveit_node);
    executor.spin();

    rclcpp::shutdown();
    return 0;
}
