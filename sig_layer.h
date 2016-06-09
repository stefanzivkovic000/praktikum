#pragma once

#include <cmath>

#include "linear_layer.h"

class sig_layer: public linear_layer {
private:
	double activation_fn(double a) {
		return 1.0 / (1.0 + exp(-a));
	}
	double activation_fn_inverse(double x) {
		return log(x / (1 - x));
	}
	double activation_fn_prime(double a) {
		return activation_fn(a) * (1.0 - activation_fn(a));
	}
	
public:
	sig_layer(int nn, int ni, int bias, double* weights, double* deltas, int my_index) :
		linear_layer(nn, ni, bias, weights, deltas, my_index) { }
};