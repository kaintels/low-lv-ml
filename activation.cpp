#include "activation.h"
#include "Eigen/src/Core/Matrix.h"


Eigen::MatrixXd Sigmoid(Eigen::MatrixXd input){
    Eigen::MatrixXd result = 1.0 / (1.0 + (-input).array().exp());
    return result;
}

Eigen::MatrixXd Tanh(Eigen::MatrixXd input){
    Eigen::MatrixXd result = (input.array().exp() - (-input).array().exp()) / (input.array().exp() + (-input).array().exp());
    return result;
}

Eigen::MatrixXd ReLU(Eigen::MatrixXd input){
    Eigen::MatrixXd result = input.array().cwiseMax(0);
    return result;
}

Eigen::MatrixXd LeakyReLU(Eigen::MatrixXd input, float negative_slope){
    Eigen::MatrixXd result = input.array().cwiseMax(0) + negative_slope * input.array().cwiseMin(0);
    return result;
}

Eigen::MatrixXd ELU(Eigen::MatrixXd input, float alpha){
    Eigen::MatrixXd result = input.array().cwiseMax(0) + alpha * (input.array().cwiseMin(0).exp() - 1);
    return result;
}

Eigen::MatrixXd Softmax(Eigen::MatrixXd input){
    Eigen::MatrixXd result = input.array().exp().colwise() / input.array().exp().rowwise().sum().array();
    return result;
}