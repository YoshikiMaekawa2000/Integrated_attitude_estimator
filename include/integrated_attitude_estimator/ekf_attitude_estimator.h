#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <geometry_msgs/Vector3Stamped.h>
#include <geometry_msgs/QuaternionStamped.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>

//Custom message
#include "integrated_attitude_estimator/EularAngle.h"


class EKFAttitudeEstimator{
    public:
        EKFAttitudeEstimator();
        ~EKFAttitudeEstimator(){};
        bool init_process();
        void imu_callback(const sensor_msgs::Imu::ConstPtr& msg);
        void dnn_angle_callback(const integrated_attitude_estimator::EularAngle::ConstPtr& msg);

        void prior_process();
        void posterior_process();

    private:
        ros::NodeHandle nh;
        ros::NodeHandle private_nh;

        ros::Subscriber imu_sub;
        ros::Subscriber angle_sub;
        ros::Publisher ekf_angle_pub;

        tf2_ros::TransformBroadcaster br;
        tf2_ros::TransformListener listener;

        integrated_attitude_estimator::EularAngle dnn_angle;
        sensor_msgs::Imu imu_data;
};