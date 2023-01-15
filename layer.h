#pragma once
#include <Eigen/Core>
#include <fstream>
#include <iostream>
#include <ostream>
#include <string>

Eigen::MatrixXd Flatten(Eigen::MatrixXd input, std::string major);

Eigen::MatrixXd Linear(std::string layer_name, Eigen::MatrixXd input, int feature, int mini_batch);