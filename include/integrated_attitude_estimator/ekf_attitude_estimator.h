#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <geometry_msgs/Vector3Stamped.h>
#include <geometry_msgs/QuaternionStamped.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/LU>

//Custom message
#include "integrated_attitude_estimator/EularAngle.h"


class EKFAttitudeEstimator{
    public:
        EKFAttitudeEstimator();
        ~EKFAttitudeEstimator();
        bool init_process();
        void imu_callback(const sensor_msgs::Imu::ConstPtr& msg);
        integrated_attitude_estimator::EularAngle get_correct_angular_velocity(sensor_msgs::Imu imu_data);
        void dnn_angle_callback(const integrated_attitude_estimator::EularAngle::ConstPtr& msg);

        void prior_process(integrated_attitude_estimator::EularAngle angular_velocity);
        void posterior_process(integrated_attitude_estimator::EularAngle dnn_angle);

        void publish_angle();

    private:
        ros::NodeHandle nh;
        ros::NodeHandle private_nh;

        ros::Subscriber imu_sub;
        ros::Subscriber angle_sub;
        ros::Publisher ekf_angle_pub;

        tf2_ros::Buffer tfBuffer;
        tf2_ros::TransformListener tfListener;
        tf2_ros::TransformBroadcaster tfBroadcaster;

        const int robot_state_size = 3; // Roll Pitch Yaw

        // DNN params
        integrated_attitude_estimator::EularAngle dnn_angle;
        double sigma_dnn = 0.01;

        // IMU params
        sensor_msgs::Imu imu_data;
        integrated_attitude_estimator::EularAngle angular_velocity;
        ros::Time imu_prev_time;
        ros::Time imu_current_time;
        double imu_duration = 0.0;
        double sigma_imu = 0.01;

        /*objects*/
		Eigen::VectorXd X;
		Eigen::MatrixXd P;
        integrated_attitude_estimator::EularAngle estimated_angle;
};