#include "neural_network.h"

int main() {

	neural_network<linear_layer, linear_layer> neural_net(1, 1);
	std::cout << neural_net.compute_output(new double(1))[0] << std::endl;

}