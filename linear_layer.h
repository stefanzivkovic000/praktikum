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
	linear_layer(int nn, int ni) : layer(nn, ni, 1) { }

	matrix* derivative_by_input(double *input) {
		return nullptr;
	}

	matrix* derivative_by_weights() {
		return nullptr;
	}
};