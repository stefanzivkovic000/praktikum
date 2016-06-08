#pragma once

#include "layer.h"
extern int nLayers;
class linear_layer : public layer {
private:
	double activation_fn(double a) {
		return a;
	}
	double activation_fn_inverse(double x) {
		return x;
	}
	double activation_fn_prime(double a) {
		return 1.0;
	}

public:
	linear_layer(int nn, int ni, int bias, double* weights, double* deltas, int my_index) :
		layer(nn, ni, bias, weights, deltas, my_index) { }
	double d_ai_xj(int i, int j, double x) {
		return weights[i * ni + j]; 
	}
	double d_ai_wij(int j) {
		return inputs[j];
	}
};