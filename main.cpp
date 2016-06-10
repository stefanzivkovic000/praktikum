#include "neural_network.h"
#include <time.h>
#include <string>

int main() {
	neural_network < tanhip_layer, sig_layer > neural_net(1, 20);
	double *arr = new double[1];
	srand(time(NULL));
	for (int i = 1; i < 1001; i++) {
		arr[0] = ((double)rand() / (double)RAND_MAX); // (0, 1)
		double res = sin(arr[0]);
		std::cout << "PASS: " << i << "; IN: " << arr[0] << 
			"; expRES: " << res << "; netOUT: " <<
			neural_net.compute_output(arr)[0] << std::endl;
		neural_net.backPropagate(res);
		std::cout << std::endl;
	}
	neural_net.printWeights();
	std::string input;
	while (getline(std::cin, input) && input[0] != '\0'){
		arr[0] = stod(input);
		double res = sin(arr[0]);
		std::cout  <<"IN: " << arr[0] <<
			 "; expRES: " << res << "; netOUT: " <<
			neural_net.compute_output(arr)[0] << std::endl;
		neural_net.backPropagate(res); 
		std::cout << std::endl;
	}
}