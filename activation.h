#include <cmath>
#include <Eigen/Core>
#include <fstream>
#include <iostream>
#include <ostream>
#include <string>


Eigen::MatrixXd Sigmoid(Eigen::MatrixXd input){

    auto m_input = -input;
    auto exp_ = m_input.array().exp();
    Eigen::MatrixXd result = 1.0 / (1.0 + exp_);

    return result;
}