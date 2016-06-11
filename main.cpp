#include "neural_network.h"
#include <time.h>
#include <string>

int main() {
	neural_network < tanhip_layer, sig_layer > neural_net(2, 10);
	double *arr = new double[2];
	srand(time(NULL));
	for (int i = 1; i < 10001; i++) {
		arr[0] = rand() % 2;
		arr[1] = rand() % 2;
		double res = (int)arr[0] ^ (int)arr[1];
		std::cout << "PASS: " << i << "; IN: " << arr[0] <<
			", " << arr[1] << "; expOUT: " << res << "; netOUT: " <<
			neural_net.train_network_pass(arr, res)[0] << std::endl;
		std::cout << std::endl;
	}
	neural_net.printWeights();
	std::string input;
	while (getline(std::cin, input) && input[0] != '\0'){
		arr[0] = input[0] - '0';
		arr[1] = input[1] - '0';
		double res = (int)arr[0] ^ (int)arr[1];
		std::cout <<"IN: " << arr[0] <<
			", " << arr[1] << "; expOUT: " << res << "; netOUT: " <<
			neural_net.train_network_pass(arr, res)[0] << std::endl; 
		std::cout << std::endl;
	}
	delete[] arr;
}