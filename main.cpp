#include "layer.h"
#include "activation.h"

int main() {

    auto inputs = Eigen::MatrixXd(2, 10);
    inputs = inputs.setOnes();

    auto x_1 = LeakyReLU(Linear("layer1", inputs, 10, 5), 0.01);
    auto result = Linear("layer2", x_1, 5, 2);

    std::cout << result;

}