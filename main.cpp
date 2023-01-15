#include "layer.h"
#include "activation.h"

int main() {

    auto inputs = Eigen::MatrixXd(2, 10);
    inputs = inputs.setRandom();

    std::cout << inputs << std::endl;

    auto result = Flatten(inputs, "row");
    // auto x_1 = ELU(Linear("layer1", inputs, 10, 5), 0.01);
    // auto result = Linear("layer2", x_1, 5, 2);

    std::cout << " " << std::endl;
    std::cout << result << std::endl;

    return 0;

}