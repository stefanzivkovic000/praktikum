#include "neural_network.h"

int main() {

	neural_network<tanhip_layer, tanhip_layer> neural_net(2, 2);
	double a[] = { 1, 2, 3, 4 ,5};

		neural_net.printWeights();
		std::cout << neural_net.compute_output(a)[0] << std::endl;
		std::cout << std::endl;
		neural_net.backPropagate(); // imala bi za parametar ocekivani izlaz mreze
		neural_net.printDeltas();
		neural_net.printWeights();

}