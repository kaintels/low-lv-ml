#include "layer.h"

int main() {

    auto inputs = Eigen::MatrixXd(2, 10);
    inputs = inputs.setOnes();

    auto x = Linear("layer1", inputs, 10, 5);
    auto result = Linear("layer2", x, 5, 2);

    std::cout << result;

}