#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <rclcpp/rclcpp.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <moveit/move_group_interface/move_group_interface.hpp>
#include <my_robot_interfaces/msg/pose_command.hpp>
#include <my_robot_interfaces/srv/get_current_pose.hpp>
#include <std_msgs/msg/string.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>

#include <yolos/yolos.hpp>
#include <opencv2/opencv.hpp>

#include <vector>
#include <cmath>
#include <atomic>
#include <memory>
#include <fstream>
#include <algorithm>
#include <limits>

using MoveGroupInterface = moveit::planning_interface::MoveGroupInterface;

using PoseCommand = my_robot_interfaces::msg::PoseCommand;
using GetCurrentPose = my_robot_interfaces::srv::GetCurrentPose;

using String = std_msgs::msg::String;
using LaserScan = sensor_msgs::msg::LaserScan;
using Image = sensor_msgs::msg::Image;

using namespace std::placeholders;

enum Classes {
  cone,
  nose,
  shoulder,
  gauge,
  rear,
  corner,
  lost
};

struct DrillBitCenter {
    bool valid;
    double center_y;
    double y_min;  // tip of drill bit (lowest y value detection)
    double y_max;  // rear of drill bit
};

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

        yolo_callback_group_ = main_node_->create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);
        
        auto sub_options = rclcpp::SubscriptionOptions();
        sub_options.callback_group = yolo_callback_group_;

        image_sub_ = main_node_->create_subscription<Image>(
            "camera/image_raw", 10,
            std::bind(&Commander::imageCallback, this, _1),
            sub_options);

        RCLCPP_INFO(main_node_->get_logger(), "Commander initialized with YOLO detector");

        // Initialize centering parameters
        z_tolerance_ = main_node_->declare_parameter<double>("centering_z_tolerance", 10.0);
        z_step_size_ = main_node_->declare_parameter<double>("centering_z_step", 0.001);  // 5mm steps
        radial_step_size_ = main_node_->declare_parameter<double>("centering_radial_step", 0.001);  // 5mm steps
        target_drill_bit_ratio_ = main_node_->declare_parameter<double>("centering_target_ratio", 0.5);  // 70% of frame height
        
        // Max radial position from URDF joint limits: lower=-0.12, upper=0.06
        max_radial_position_ = 0.06;
        
        centering_enabled_ = false;
        
        RCLCPP_INFO(main_node_->get_logger(), 
            "Centering params: z_tol=%.1f, z_step=%.4f, radial_step=%.4f, max_radial=%.3f, target_ratio=%.2f",
            z_tolerance_, z_step_size_, radial_step_size_, max_radial_position_, target_drill_bit_ratio_);
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

    void startCentering() {
        centering_enabled_ = true;
        auto current_pose = arm_group_->getCurrentPose();
        RCLCPP_INFO(main_node_->get_logger(), 
            "Centering started. Current radial: %.4f, Max radial: %.4f", 
            current_pose.pose.position.x, max_radial_position_);
    }

    void stopCentering() {
        centering_enabled_ = false;
        RCLCPP_INFO(main_node_->get_logger(), "Centering stopped");
    }

    bool isCenteringEnabled() const { return centering_enabled_; }

private:

    void initYoloDetector() {
        std::string model_path = main_node_->declare_parameter<std::string>(
            "yolo_model_path", ament_index_cpp::get_package_share_directory("cam_auto_adjusting") + "/models" + "/yolo26m_cam_one.onnx");
        int num_classes = main_node_->declare_parameter<int>("yolo_num_classes", 7);
        bool use_gpu = main_node_->declare_parameter<bool>("yolo_use_gpu", false);
        
        conf_threshold_ = main_node_->declare_parameter<double>("yolo_conf_threshold", 0.05);
        iou_threshold_ = main_node_->declare_parameter<double>("yolo_iou_threshold", 0.05);
    

        std::string labels_path = "/tmp/yolo_labels.names";

        try {
            yolo_detector_ = std::make_unique<yolos::YOLODetector>(
                model_path, labels_path, use_gpu);
            
            class_names_ = yolo_detector_->getClassNames();

            RCLCPP_INFO(main_node_->get_logger(), 
                "YOLO detector initialized: model=%s, num_classes=%d, GPU=%s, conf_threshold: %.2f, iou_threshold: %.2f",
                model_path.c_str(), num_classes, use_gpu ? "true" : "false", conf_threshold_, iou_threshold_);
        } catch (const std::exception& e) {
            RCLCPP_ERROR(main_node_->get_logger(), 
                "Failed to initialize YOLO detector: %s", e.what());
        }
    }

    DrillBitCenter findDrillBitCenter(const std::vector<yolos::Detection>& detections) {
        DrillBitCenter result{false, 0.0, 0.0, 0.0};
        
        if (detections.empty()) {
            return result;
        }

        double y_min = std::numeric_limits<double>::max();
        double y_max = std::numeric_limits<double>::lowest();
        bool found_rear = false;
        bool found_tip = false;

        for (const auto& detection : detections) {
            // Get bounding box coordinates (x, y, width, height -> convert to y_min, y_max)
            double det_y_min = detection.box.y;
            double det_y_max = detection.box.y + detection.box.height;

            // The rear or corner detection gives us y_max (bottom of drill bit in image)
            if (detection.classId == Classes::rear || detection.classId == Classes::corner) {
                if (det_y_max > y_max) {
                    y_max = det_y_max;
                }
                found_rear = true;
            }

            // Track the detection with minimum y value (tip of drill bit)
            if (det_y_min < y_min) {
                y_min = det_y_min;
                found_tip = true;
            }
        }

        if (found_rear && found_tip) {
            result.valid = true;
            result.y_min = y_min;
            result.y_max = y_max;
            result.center_y = (y_min + y_max) / 2.0;
        }

        return result;
    }

    void moveArmZ(double delta_z) {
        auto current_pose = arm_group_->getCurrentPose();
        double new_z = current_pose.pose.position.z + delta_z;
        
        RCLCPP_INFO(main_node_->get_logger(), "Moving Z: %.4f -> %.4f (delta: %.4f)",
            current_pose.pose.position.z, new_z, delta_z);
        
        goToPositionTarget(
            current_pose.pose.position.x,
            current_pose.pose.position.y,
            new_z,
            true  // use cartesian path for smooth motion
        );
    }

    void moveArmRadial(double delta_y) {
        auto current_pose = arm_group_->getCurrentPose();
        double new_y = current_pose.pose.position.y + delta_y;
        
        RCLCPP_INFO(main_node_->get_logger(), "Moving radially (Y-axis): %.4f -> %.4f (delta: %.4f, max: %.4f)",
            current_pose.pose.position.y, new_y, delta_y, max_radial_position_);
        
        goToPositionTarget(
            current_pose.pose.position.x,
            new_y,
            current_pose.pose.position.z,
            true  // use cartesian path for smooth motion
        );
    }

    void putText(cv::Mat& frame, const std::string& text, const cv::Point& org) {
        cv::putText(frame, text, org, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
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

            int frame_height = frame.rows;
            int frame_width = frame.cols;
            int frame_center_y = frame_height / 2;

            if (!detections.empty()) {
                RCLCPP_INFO(main_node_->get_logger(), 
                    "YOLO detected %zu objects", detections.size());
                
                yolo_detector_->drawDetections(frame, detections);

                // Find drill bit center
                DrillBitCenter drill_bit = findDrillBitCenter(detections);

                if (drill_bit.valid) {
                    double diff = frame_center_y - drill_bit.center_y;
                    double drill_bit_length = drill_bit.y_max - drill_bit.y_min;
                    double drill_bit_ratio = drill_bit_length / frame_height;

                    // Draw visualization
                    cv::circle(frame, cv::Point(frame_width / 2, frame_center_y), 4, cv::Scalar(255, 255, 0), -1);  // frame center
                    cv::circle(frame, cv::Point(frame_width / 2, static_cast<int>(drill_bit.center_y)), 4, cv::Scalar(255, 0, 255), -1);  // drill bit center
                    cv::line(frame, cv::Point(frame_width / 2, static_cast<int>(drill_bit.y_min)), 
                             cv::Point(frame_width / 2, static_cast<int>(drill_bit.y_max)), cv::Scalar(0, 255, 255), 2);
                    
                    // Get current radial position (Y-axis)
                    auto current_pose = arm_group_->getCurrentPose();
                    double current_radial = current_pose.pose.position.y;

                    // Display info
                    putText(frame, "Ratio: " + std::to_string(drill_bit_ratio * 100).substr(0, 5) + "%", cv::Point(10, 20));
                    putText(frame, "Diff: " + std::to_string(diff).substr(0, 6), cv::Point(10, 40));
                    // putText(frame, "Radial: " + std::to_string(current_radial).substr(0, 6) + "/" + std::to_string(max_radial_position_).substr(0, 5), cv::Point(10, 60));

                    if (centering_enabled_) {
                        // Centering logic
                        if (diff < 0 && std::abs(diff) > z_tolerance_) {
                            // Drill bit center is below frame center -> move down
                            putText(frame, "Moving DOWN (Z-axis)", cv::Point(frame_width / 2 - 80, 100));
                            RCLCPP_INFO(main_node_->get_logger(), "Moving down, diff=%.2f", diff);
                            moveArmZ(-z_step_size_);
                        }
                        else if (diff > 0 && std::abs(diff) > z_tolerance_) {
                            // Drill bit center is above frame center -> move up
                            putText(frame, "Moving UP (Z-axis)", cv::Point(frame_width / 2 - 80, 100));
                            RCLCPP_INFO(main_node_->get_logger(), "Moving up, diff=%.2f", diff);
                            moveArmZ(z_step_size_);
                        }
                        else {
                            // Z-axis centered, now check radial movement
                            if (drill_bit_ratio >= target_drill_bit_ratio_) {
                                // Target ratio reached - stop centering
                                putText(frame, "CENTERED - Target ratio reached!", cv::Point(frame_width / 2 - 100, 100));
                                RCLCPP_INFO(main_node_->get_logger(), 
                                    "Centering complete! Drill bit ratio: %.2f%% >= %.2f%%",
                                    drill_bit_ratio * 100, target_drill_bit_ratio_ * 100);
                                centering_enabled_ = false;
                            }
                            // else if (current_radial + radial_step_size_ >= max_radial_position_) {
                            //     // Max radial position reached - stop centering
                            //     putText(frame, "STOPPED - Max radial reached", cv::Point(frame_width / 2 - 100, 100));
                            //     RCLCPP_WARN(main_node_->get_logger(), 
                            //         "Max radial position reached (%.4f >= %.4f). Stopping. Ratio: %.2f%%",
                            //         current_radial, max_radial_position_, drill_bit_ratio * 100);
                            //     centering_enabled_ = false;
                            // }
                            else {
                                // Move forward radially
                                putText(frame, "Moving FORWARD (Radial)", cv::Point(frame_width / 2 - 80, 100));
                                RCLCPP_INFO(main_node_->get_logger(), 
                                    "Moving radially, ratio=%.2f%%, current_y=%.4f",
                                    drill_bit_ratio * 100, current_radial);
                                moveArmRadial(radial_step_size_);
                            }
                        }
                    } else {
                        // putText(frame, "Centering DISABLED", cv::Point(frame_width / 2 - 80, 100));
                    }
                } else {
                    putText(frame, "Cannot detect drill bit center", cv::Point(frame_width / 2 - 100, 100));
                    RCLCPP_WARN(main_node_->get_logger(), "Could not find drill bit center (need rear/corner + tip)");
                }

                cv::imshow("detections", frame);
                cv::waitKey(1);
            } else {
                putText(frame, "No detections", cv::Point(frame_width / 2 - 50, 100));
                cv::imshow("detections", frame);
                cv::waitKey(1);
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

    rclcpp::CallbackGroup::SharedPtr yolo_callback_group_;

    std::unique_ptr<yolos::YOLODetector> yolo_detector_;
    double conf_threshold_{0.4};
    double iou_threshold_{0.45};

    std::vector<std::string> class_names_;
    
    // Centering parameters
    double z_tolerance_{15.0};           // pixels
    double z_step_size_{0.001};          // meters
    double radial_step_size_{0.001};     // meters
    double max_radial_position_{0.5};    // absolute max X position in meters (from joint limits)
    double target_drill_bit_ratio_{0.5}; // percentage of frame height
    
    // Centering state
    bool centering_enabled_{false};
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
