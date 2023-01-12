#include <Eigen/Core>
#include <fstream>
#include <iostream>
#include <ostream>

auto Dense(Eigen::MatrixXd input, Eigen::MatrixXd weight, Eigen::MatrixXd bias){

    auto result = Eigen::VectorXd(bias.size());
    auto dot_product = Eigen::VectorXd(bias.size());
    for (int i = 0; i < dot_product.size(); ++i) {
        dot_product[i] = input.cwiseProduct(weight).sum();
    }

    result = dot_product + bias;
    return result;
}


int main() {
    float number;
    std::ifstream awfile("aw.txt");
    std::ifstream abfile("ab.txt");

    auto weight = Eigen::VectorXd(10);
    auto bias = Eigen::VectorXd(1);
    auto inputs = Eigen::VectorXd(10);
    inputs = inputs.setOnes();

    for (int i = 0; i < weight.size(); ++i) {
        awfile >> number;
        weight[i] = number;
    }

    for (int i = 0; i < bias.size(); ++i) {
        abfile >> number;
        bias[i] = number;
    }

    auto result = Dense(inputs, weight, bias);

    std::cout << result;

}



