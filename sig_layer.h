#pragma once

#include <cmath>

#include "linear_layer.h"

class sig_layer: public linear_layer {
private:
	double activation_fn(double a) {
		return 1.0 / (1.0 + exp(-a));
	}

	double activation_fn_prime(double a) {
		return activation_fn(a) * (1.0 - activation_fn(a));
	}
public:
	sig_layer(int nn, int ni) : linear_layer(nn, ni) { }

	matrix* derivative_by_input(double *input) {
		return nullptr;
	}

	matrix* derivative_by_weights() {
		return nullptr;
	}
};