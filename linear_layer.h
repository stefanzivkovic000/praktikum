#pragma once

#include "layer.h"

class linear_layer: public layer {
private:
	double activation_fn(double a) {
		return a;
	}

	double activation_fn_prime(double a) {
		return 1.0;
	}
public:
	linear_layer(int nn, int ni, int bias, double* weights, double* deltas, int my_index) :
		layer(nn, ni, bias, weights, deltas, my_index) { }
};