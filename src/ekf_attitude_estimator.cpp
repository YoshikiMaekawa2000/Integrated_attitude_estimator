#include "integrated_attitude_estimator/ekf_attitude_estimator.h"

EKFAttitudeEstimator::EKFAttitudeEstimator():private_nh("~"), tfListener(tfBuffer){
    //Subscribers
    imu_sub = nh.subscribe("/imu/data", 1, &EKFAttitudeEstimator::imu_callback, this);
    angle_sub = nh.subscribe("/dnn_angle", 1, &EKFAttitudeEstimator::dnn_angle_callback, this);
    //Publishers
    ekf_angle_pub = nh.advertise<integrated_attitude_estimator::EularAngle>("ekf_angle", 1);

    X = Eigen::VectorXd::Zero(robot_state_size);
	const double initial_sigma = 1.0e-100;
	P = initial_sigma*Eigen::MatrixXd::Identity(robot_state_size, robot_state_size);

    bool init_result = init_process();
    if(!init_result){
        ROS_ERROR("Failed to initialize EKF attitude estimator");
        exit(1);
    }
}

bool EKFAttitudeEstimator::init_process(){
    bool result = true;


    return result;
}

EKFAttitudeEstimator::~EKFAttitudeEstimator(){}

void EKFAttitudeEstimator::imu_callback(const sensor_msgs::Imu::ConstPtr& msg){
    imu_data = *msg;
    imu_prev_time = imu_current_time;
    imu_current_time = imu_data.header.stamp;
    imu_duration = (imu_current_time - imu_prev_time).toSec();
    
    if(imu_duration > 0.5){
        imu_duration = 0.0;
    }
    
    if(use_imu_angular_velocity==true){
        angular_velocity = get_correct_angular_velocity(imu_data);
        prior_process(angular_velocity, sigma_imu_velocity);
    }

    if(use_imu_angle==true){
        imu_angle = get_imu_angle(imu_data);
        posterior_process(imu_angle, sigma_imu_angle);
    }

    if(use_imu_angular_velocity==true || use_imu_angle==true){
        publish_angle();
    }
}

integrated_attitude_estimator::EularAngle EKFAttitudeEstimator::get_imu_angle(sensor_msgs::Imu imu_data){
    integrated_attitude_estimator::EularAngle angle;
    double imu_roll, imu_pitch, imu_yaw;
    tf::Quaternion q;
    quaternionMsgToTF(imu_data.orientation, q);
    tf::Matrix3x3(q).getRPY(imu_roll, imu_pitch, imu_yaw);

    angle.roll = imu_pitch * -1.0;
    angle.pitch = (imu_roll + M_PI/2.0) * 1.0;
    angle.yaw = imu_yaw;

    return angle;
}

integrated_attitude_estimator::EularAngle EKFAttitudeEstimator::get_correct_angular_velocity(sensor_msgs::Imu imu_data){
    integrated_attitude_estimator::EularAngle velocity;

    // TODO: Correct angular velocity
    velocity.roll = imu_data.angular_velocity.z;
    velocity.pitch = -1.0 * imu_data.angular_velocity.x;
    velocity.yaw = -1.0 * imu_data.angular_velocity.y;

    return velocity;
}

void EKFAttitudeEstimator::dnn_angle_callback(const integrated_attitude_estimator::EularAngle::ConstPtr& msg){
    dnn_angle = *msg;
    dnn_angle.roll = dnn_angle.roll * M_PI / 180.0;
    dnn_angle.pitch = dnn_angle.pitch * M_PI / 180.0;
    dnn_angle.yaw = dnn_angle.yaw * M_PI / 180.0;

    if(use_dnn_angle==true){
        posterior_process(dnn_angle, sigma_dnn_angle);
        publish_angle();
    }
}

void EKFAttitudeEstimator::prior_process(integrated_attitude_estimator::EularAngle angular_velocity, double sigma){
    printf("prior process\n");
    double roll = X(0);
    double pitch = X(1);
    double yaw = X(2); // 0.0

    double delta_r = angular_velocity.roll * imu_duration;
    double delta_p = angular_velocity.pitch * imu_duration;
    double delta_y = angular_velocity.yaw * imu_duration;
    Eigen::Vector3d Drpy = {delta_r, delta_p, delta_y};

    Eigen::Matrix3d Rot_rpy;	//normal rotation
	Rot_rpy <<	1,	sin(roll)*tan(pitch),	cos(roll)*tan(pitch),
			0,	cos(roll),		-sin(roll),
			0,	sin(roll)/cos(pitch),	cos(roll)/cos(pitch);

    Eigen::VectorXd F(X.size());
    // F <<	roll,
    //         pitch,
    //         yaw;
    
    F(0) = roll;
    F(1) = pitch;
    F(2) = yaw;
    
    F = F + Rot_rpy*Drpy;

    /*jF*/
	Eigen::MatrixXd jF = Eigen::MatrixXd::Zero(X.size(), X.size());
    jF(0, 0) = 1 + (cos(roll)*tan(pitch)*delta_p + sin(roll)*tan(pitch)*delta_y);
    jF(0, 1) = sin(roll)/cos(pitch)/cos(pitch)*delta_p + cos(roll)/cos(pitch)/cos(pitch)*delta_y;
    jF(0, 2) = 0.0;
    jF(1, 0) = -sin(roll)*delta_p + cos(roll)*delta_y;
    jF(1, 1) = 1.0;
    jF(1, 2) = 0.0;
    jF(2, 0) = cos(roll)/cos(pitch)*delta_p - sin(roll)/cos(pitch)*delta_y;
    jF(2, 1) = sin(roll)*sin(pitch)/cos(pitch)/cos(pitch)*delta_p + cos(roll)*sin(pitch)/cos(pitch)/cos(pitch)*delta_y;
    jF(2, 2) = 1.0;

    Eigen::MatrixXd Q = sigma*Eigen::MatrixXd::Identity(X.size(), X.size());

    /*Update*/
	X = F;
    X(2) = 0.0;
	P = jF*P*jF.transpose() + Q;
}

void EKFAttitudeEstimator::posterior_process(integrated_attitude_estimator::EularAngle angle, double sigma){
    printf("posterior process\n");

    Eigen::VectorXd Z(3);
	Z <<	angle.roll,
		angle.pitch,
		angle.yaw;

    Eigen::VectorXd Zp = X;
    Eigen::MatrixXd jH = Eigen::MatrixXd::Identity(Z.size(), X.size());
	Eigen::VectorXd Y = Z - Zp;
	Eigen::MatrixXd R = sigma*Eigen::MatrixXd::Identity(Z.size(), Z.size());
	Eigen::MatrixXd S = jH*P*jH.transpose() + R;
	Eigen::MatrixXd K = P*jH.transpose()*S.inverse();
	X = X + K*Y;

    X(2) = 0.0;

    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(X.size(), X.size());
	P = (I - K*jH)*P;
}

void EKFAttitudeEstimator::publish_angle(){
    estimated_angle.header.stamp = ros::Time::now();
    estimated_angle.roll = X(0);
    estimated_angle.pitch = X(1);
    estimated_angle.yaw = 0.0;

    ekf_angle_pub.publish(estimated_angle);
}

int main(int argc, char **argv){
    ros::init(argc, argv, "ekf_attitude_estimator");
	std::cout << "EKF Attitude Estimator" << std::endl;
    EKFAttitudeEstimator ekf_attitude_estimator;

    ros::spin();

    return 0;
}