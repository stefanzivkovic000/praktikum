#include "neural_network.h"
#include <time.h>
#include <string>
int main() {

	neural_network<sig_layer, sig_layer> neural_net(2, 20);
	double *arr = new double[2];
	srand(time(NULL));
	for (int i = 1; i < 1001; i++) {
		arr[0] = rand() % 2;
		arr[1] = rand() % 2;
		double res = (int) arr[0] ^ (int) arr[1];
		std::cout << "PASS: " << i << "; IN: " << arr[0] << 
			", " << arr[1] << "; RES: " << res << "; OUT: " <<
			neural_net.compute_output(arr)[0] << std::endl;
		neural_net.backPropagate(res);
		std::cout << std::endl;
	}
	std::string input; // format za ulaz je:x y
	while (getline(std::cin, input) && input[0] != '\0'){
		arr[0] = input[0] - '0';
		arr[1] = input[2] - '0';
		double res = (int)arr[0] ^ (int)arr[1];
		std::cout  <<"IN: " << arr[0] <<
			", " << arr[1] << "; RES: " << res << "; OUT: " <<
			neural_net.compute_output(arr)[0] << std::endl;
		neural_net.backPropagate(res); 
		std::cout << std::endl;
	}
}