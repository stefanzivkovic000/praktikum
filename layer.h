#pragma once

#include <iostream>

class layer {
protected:
	int my_index;		//indeks sloja u mrezi - skriveni sloj - 0, izlazni sloj - 1
	int nn;			//broj neurona u sloju
	int ni;			//broj ulaza
	int bias;		//1 ako ima bias, 0 inace								
	double* weights;
	double* potentials;
	double* output;
	double* deltas;
	virtual double activation_fn(double a) = 0;			//a je potencijal
	virtual double activation_fn_prime(double a) = 0;		//a je potencijal
public:
	layer(int nn, int ni, int bias, double* weights, double* deltas, int my_index) {
		
		this -> nn = nn;
		this -> ni = ni;
		this -> bias = bias;
		this -> weights = weights;
		this -> output = new double[nn];
		this -> my_index = my_index;

		potentials = new double[nn];

	}
	~layer() {
		delete [] potentials;
		delete [] output;
	}
	void compute_potentials(double *input) {			

		for (int i = 0; i < nn; i++) {
			potentials[i] = (bias) ? 1 * weights[i * (ni + 1)] : 0.0;
			for (int j = 0; j < ni; j++) {
				potentials[i] += input[j] * weights[i * (ni + 1) + j + 1];
			}
		}

	}

	double* compute_output(double *input) {
		
		compute_potentials(input);

		for (int i = 0; i < nn; i++)
			output[i] = activation_fn(potentials[i]);

		return output;

	}

	void update_weights(double *new_weights) {
		int to = (ni + 1) * nn;
		for (int i = 0; i < to; i++) 
			weights[i] = new_weights[i];
	}

	int get_nn() {
		return nn;
	}

	int get_ni() {
		return ni;
	}

	double* get_potentials() {
		return potentials;
	}

	double* get_output() {
		return output;
	}

	double* get_deltas() {
		return deltas;
	}

};