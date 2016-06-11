#pragma once

#include "linear_layer.h"

class tanhip_layer: public linear_layer {
private:
	double activation_fn(double a) {
		return (exp(a) - exp(-a)) / (exp(a) + exp(-a));
	}
	double activation_fn_prime(double a) {
		return 1.0 - activation_fn(a) * activation_fn(a);
	}
public:
	tanhip_layer(int nn, int ni, int bias, double* weights, double* deltas, layer* next_layer=nullptr) :
		linear_layer(nn, ni, bias, weights, deltas, next_layer) { }
};