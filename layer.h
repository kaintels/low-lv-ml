#include <Eigen/Core>
#include <fstream>
#include <iostream>
#include <ostream>
#include <string>

Eigen::MatrixXd Flatten(Eigen::MatrixXd input, std::string major="row"){

    if (major == "row"){
        auto result = input.reshaped<Eigen::RowMajor>().transpose();
        return result;
    }

    else {
        auto result = input.reshaped().transpose();
        return result;
    }
}

Eigen::MatrixXd Linear(std::string layer_name, Eigen::MatrixXd input, int feature, int mini_batch){

    float number;

    std::ifstream awfile(layer_name + ".weight.txt");
    std::ifstream abfile(layer_name + ".bias.txt");

    auto weight = Eigen::MatrixXd(mini_batch, feature);
    auto bias = Eigen::VectorXd(mini_batch);

    for (int j = 0; j < weight.rows(); ++j){
        for (int i = 0; i < weight.cols(); ++i) {
            awfile >> number;
            weight(j, i) = number;
        }
    }

    for (int i = 0; i < bias.size(); ++i) {
        abfile >> number;
        bias(i) = number;
    }
    
    auto dot_product = input.lazyProduct(weight.transpose());
    auto result = Eigen::MatrixXd(dot_product.rows(), dot_product.cols());

    for (int j = 0; j < dot_product.rows(); ++j){
        result(j, Eigen::all) = dot_product(j, Eigen::all).transpose() + bias;
    }

    return result;
}

