#pragma once

#include "linear_layer.h"
#include "tanhip_layer.h"
#include "sig_layer.h"

template<typename H, typename O>
class neural_network {
protected:			//za slucaj da je prosirujemo
	H *hidden_layer;
	O *output_layer;
	int ni;			//broj ulaza u mrezu
	double* output;
public:
	neural_network(int in_num, int hl_num, int ol_num = 1) {
		hidden_layer = new H(hl_num, in_num);
		output_layer = new O(ol_num, hl_num, 0);
	}

	double* compute_output(double *input) {
		output = output_layer -> compute_output(hidden_layer -> compute_output(input));
		return output;
	}

	double* get_output() {
		return output;
	}
};