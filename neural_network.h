#pragma once

#include <random>

#include "linear_layer.h"
#include "tanhip_layer.h"
#include "sig_layer.h"

const double min_ran = -0.1;
const double max_ran = 0.1;

template<typename H, typename O>
class neural_network {
protected:					
	H *hidden_layer;
	O *output_layer;
	double* weights;				//tezine cele mreze
	double* output;
	double* deltas;
public:
	neural_network(int in_num, int hl_num, int ol_num = 1) {
		int weights_size = (in_num + 1) * hl_num + (hl_num + 1) * ol_num;

		output = nullptr;
		weights = new double[weights_size];
		deltas = new double[weights_size];
		hidden_layer = new H(hl_num, in_num, 1, weights, deltas, 0);
		output_layer = new O(ol_num, hl_num, 1, weights + (in_num + 1) * hl_num, deltas + hl_num + 1, 1);

		std::random_device device;
		std::default_random_engine engine(device());
		std::uniform_real_distribution<double> distribution(min_ran, max_ran);

		for (int i = 0; i < weights_size; i++) {
				weights[i] = distribution(engine);
		}
	}
	
	virtual ~neural_network() {
		delete hidden_layer;
		delete output_layer;
		delete [] weights;
		delete [] deltas;
	}

	virtual double* compute_output(double *input) {     //da bi mogla da se predefinise
		output = output_layer -> compute_output(hidden_layer -> compute_output(input));
		return output;
	}

	double* get_output() {
		return output;
	}
};