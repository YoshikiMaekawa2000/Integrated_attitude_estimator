#include "integrated_attitude_estimator/ekf_attitude_estimator.h"

EKFAttitudeEstimator::EKFAttitudeEstimator(): private_nh("~"){
    //Subscribers
    imu_sub = nh.subscribe("/imu/data", 1, &EKFAttitudeEstimator::imu_callback, this);
    angle_sub = nh.subscribe("/dnn_angle", 1, &EKFAttitudeEstimator::dnn_angle_callback, this);
    //Publishers
    ekf_angle_pub = nh.advertise<integrated_attitude_estimator::EularAngle>("ekf_angle", 1);

    bool init_result = init_process();
    if(!init_result){
        ROS_ERROR("Failed to initialize EKF attitude estimator");
        exit(1);
    }
}

bool init_process(){
    bool result = true;


    return result;
}

EFKAttitudeEstimator::~EKFAttitudeEstimator(){}

void EKFAttitudeEstimator::imu_callback(const sensor_msgs::Imu::ConstPtr& msg){
    imu_data = *msg;
    prior_process();
}

void EKFAttitudeEstimator::dnn_angle_callback(const integrated_attitude_estimator::EularAngle::ConstPtr& msg){
    dnn_angle = *msg;
    posterior_process();
}

void EKFAttitudeEstimator::prior_process(){
    printf("prior process\n");


}

void EKFAttitudeEstimator::posterior_process(){
    printf("posterior process\n");


}

int main(int argc, char **argv){
    ros::init(argc, argv, "ekf_attitude_estimator");
	std::cout << "EKF Attitude Estimator" << std::endl;
    EKFAttitudeEstimator ekf_attitude_estimator;

    ros::spin();

    return 0;
}