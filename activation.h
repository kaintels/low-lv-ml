#pragma once
#include <cmath>
#include <Eigen/Core>
#include <fstream>
#include <iostream>
#include <ostream>
#include <string>

Eigen::MatrixXd Sigmoid(Eigen::MatrixXd input);

Eigen::MatrixXd Tanh(Eigen::MatrixXd input);

Eigen::MatrixXd ReLU(Eigen::MatrixXd input);

Eigen::MatrixXd LeakyReLU(Eigen::MatrixXd input, float negative_slope);

Eigen::MatrixXd ELU(Eigen::MatrixXd input, float alpha);

Eigen::MatrixXd Softmax(Eigen::MatrixXd input);