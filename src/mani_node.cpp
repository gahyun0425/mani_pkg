#define _USE_MATH_DEFINES
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <cmath>
#include <limits>
#include <stdexcept>
#include "rclcpp/rclcpp.hpp"

using Eigen::Matrix4d;
using Eigen::MatrixXd;
using Eigen::Vector3d;
using Eigen::VectorXd;
using Eigen::JacobiSVD;
using Eigen::ComputeFullU;
using Eigen::ComputeFullV;

double L1 = 26.7, L2 = 28.949, L3 = 7.75, L4 = 34.25, L5 = 7.6, L6 = 9.7;
double theta_offset2 = -1.3849;
double theta_offset3 = 1.3849;

// Convert degrees to radians
double degToRad(double degree) {
    return degree * M_PI / 180.0;
}

Matrix4d DHMatrix(double a, double alpha, double d, double theta) {
    Matrix4d T;
    T << cos(theta), -sin(theta) * cos(alpha), sin(theta)* sin(alpha), a* cos(theta),
        sin(theta), cos(theta)* cos(alpha), -cos(theta) * sin(alpha), a* sin(theta),
        0, sin(alpha), cos(alpha), d,
        0, 0, 0, 1;
    return T;
}

Matrix4d forwardKinematics(VectorXd& theta) {
    Matrix4d T1 = DHMatrix(0, 0, L1, theta(0));
    Matrix4d T2 = T1 * DHMatrix(0, -M_PI / 2, 0, theta(1) + theta_offset2);
    Matrix4d T3 = T2 * DHMatrix(L2, 0, 0, theta(2) + theta_offset3);
    Matrix4d T4 = T3 * DHMatrix(L3, -M_PI, L4, theta(3));
    Matrix4d T5 = T4 * DHMatrix(0, M_PI / 2, 0, theta(4));
    Matrix4d T6 = T5 * DHMatrix(L5, -M_PI / 2, L6, theta(5));
    return T6;
}

MatrixXd computeJacobian(VectorXd& theta) {
    MatrixXd J(6, 6);
    Matrix4d T1 = DHMatrix(0, 0, L1, theta(0));
    Matrix4d T2 = T1 * DHMatrix(0, -M_PI / 2, 0, theta(1) + theta_offset2);
    Matrix4d T3 = T2 * DHMatrix(L2, 0, 0, theta(2) + theta_offset3);
    Matrix4d T4 = T3 * DHMatrix(L3, -M_PI, L4, theta(3));
    Matrix4d T5 = T4 * DHMatrix(0, M_PI / 2, 0, theta(4));
    Matrix4d T6 = T5 * DHMatrix(L5, -M_PI / 2, L6, theta(5));

    Vector3d Oe = T6.block<3, 1>(0, 3);
    Vector3d Z0(0, 0, 1);
    Vector3d Z1 = T1.block<3, 1>(0, 2);
    Vector3d Z2 = T2.block<3, 1>(0, 2);
    Vector3d Z3 = T3.block<3, 1>(0, 2);
    Vector3d Z4 = T4.block<3, 1>(0, 2);
    Vector3d Z5 = T5.block<3, 1>(0, 2);

    J.block<3, 1>(0, 0) = Z0.cross(Oe);
    J.block<3, 1>(0, 1) = Z1.cross(Oe - T1.block<3, 1>(0, 3));
    J.block<3, 1>(0, 2) = Z2.cross(Oe - T2.block<3, 1>(0, 3));
    J.block<3, 1>(0, 3) = Z3.cross(Oe - T3.block<3, 1>(0, 3));
    J.block<3, 1>(0, 4) = Z4.cross(Oe - T4.block<3, 1>(0, 3));
    J.block<3, 1>(0, 5) = Z5.cross(Oe - T5.block<3, 1>(0, 3));

    J.block<3, 1>(3, 0) = Z0;
    J.block<3, 1>(3, 1) = Z1;
    J.block<3, 1>(3, 2) = Z2;
    J.block<3, 1>(3, 3) = Z3;
    J.block<3, 1>(3, 4) = Z4;
    J.block<3, 1>(3, 5) = Z5;

    return J;
}

MatrixXd dampedPseudoInverse(const MatrixXd& J) {
    JacobiSVD<MatrixXd> svd(J, ComputeFullU | ComputeFullV);
    VectorXd singular_values = svd.singularValues();
    MatrixXd U = svd.matrixU();
    MatrixXd V = svd.matrixV();

    double lambda2 = 0.03;

    for (long i = 0; i < singular_values.size(); ++i) {
        singular_values(i) = singular_values(i) / (singular_values(i) * singular_values(i) + lambda2);
    }

    return V * singular_values.asDiagonal() * U.transpose();
}

class ManipulatorNode : public rclcpp::Node {
public:
    ManipulatorNode() : Node("manipulator_node") {
        // 초기 관절 각도 설정
        VectorXd theta(6);
        theta << 0, -M_PI / 4, M_PI / 6, 0, M_PI / 4, -M_PI / 6;

        // 초기 end-effector 위치 계산
        Matrix4d initial_transform = forwardKinematics(theta);
        Vector3d initial_position = initial_transform.block<3, 1>(0, 3);
        RCLCPP_INFO(this->get_logger(), "Initial end-effector position: [%.4f, %.4f, %.4f]",
                    initial_position(0), initial_position(1), initial_position(2));

        // 목표 위치 설정
        Vector3d target_position(50.0, 30.0, 20.0);
        Vector3d angular_velocity(0.1, 0.1, 0.1);

        // 역기구학 계산
        VectorXd joint_angles = inverseKinematics(target_position, angular_velocity);
        RCLCPP_INFO(this->get_logger(), "Calculated joint angles (radians): [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f]", 
                    joint_angles(0), joint_angles(1), joint_angles(2), 
                    joint_angles(3), joint_angles(4), joint_angles(5));

        Matrix4d final_transform = forwardKinematics(joint_angles);
        Vector3d final_position = final_transform.block<3, 1>(0, 3);
        RCLCPP_INFO(this->get_logger(), "Calculated final position from forward kinematics: [%.4f, %.4f, %.4f]", 
                    final_position(0), final_position(1), final_position(2));
    }


private:
    VectorXd inverseKinematics(const Vector3d& position, const Vector3d& angular_velocity) {
        VectorXd theta(6);
        theta << 0, -M_PI / 4, M_PI / 6, 0, M_PI / 4, -M_PI / 6;

        Vector3d target_position = position;
        VectorXd error(6);

        for (int attempt = 0; attempt < 1000; ++attempt) {
            Matrix4d T6 = forwardKinematics(theta);
            Vector3d current_position = T6.block<3, 1>(0, 3);
            Vector3d position_error = target_position - current_position;

            MatrixXd J = computeJacobian(theta);
            error.head<3>() = position_error;
            error.tail<3>() = angular_velocity;

            MatrixXd J_pseudo_inv = dampedPseudoInverse(J);
            VectorXd delta_theta = J_pseudo_inv * error;
            theta += delta_theta;

            if (position_error.norm() < 1e-3) break;
        }
        return theta;
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ManipulatorNode>());
    rclcpp::shutdown();
    return 0;
}
