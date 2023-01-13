#include "layer.h"
#include "activation.h"

int main() {

    auto inputs = Eigen::MatrixXd(2, 10);
    inputs = inputs.setOnes();

    auto x_1 = Linear("layer1", inputs, 10, 5);
    auto x_2 = Sigmoid(x_1);
    auto result = Linear("layer2", x_2, 5, 2);

    std::cout << result;

}