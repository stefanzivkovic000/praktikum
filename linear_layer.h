#pragma once

#include "layer.h"
extern int nLayers;
class linear_layer : public layer {
private:
	virtual double activation_fn(double a) {
		return a;
	}
	virtual double activation_fn_prime(double a) {
		return 1.0;
	}

public:
	linear_layer(int nn, int ni, int bias, double* weights, double* deltas, layer* next_layer=nullptr) :
		layer(nn, ni, bias, weights, deltas, next_layer) { }
	double d_ai_xj(int i, int j) {
		return weights[i * (ni+bias) + j]; 
	}
	double d_ai_wij(int j) {
		return inputs[j];
	}
};