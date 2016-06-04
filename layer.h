#pragma once

#include <random>

#include "matrix.h"

class layer {
protected:
	int nn;			//broj neurona u sloju
	int ni;			//broj ulaza
	int bias;		//1 ako ima bias, 0 inace								
	matrix weights;		//i-ti red matrice su tezine i-tog neurona u sloju			
	double *output;
	virtual double activation_fn(double a) = 0;			//a je potencijal
	virtual double activation_fn_prime(double a) = 0;		//a je potencijal
public:
	layer(int nn, int ni, int bias) {
		
		//setvovanje broja neurona, broja ulaza, postojanja bias-a, 
		//zauzimanje prostora za tezine i izlaz
		this -> nn = nn;
		this -> ni = ni;
		this -> bias = bias;							
		weights.resize(nn, ni + bias);			//jedan neuron vise za bias
		output = new double[nn];

		//definisanje generatora slucajnih brojeva
		std::random_device device;
		std::default_random_engine engine(device());
		std::uniform_real_distribution<double> distribution(-0.1, 0.1);	//opseg je hardkodiran

		//popunjavanje tezina slucajnim vrednostima
		for (int i = 0; i < nn; i++) {
			for (int j = 0; j < ni + bias; j++)
				weights[i][j] = distribution(engine);
		}

	}

	void compute_potentials(double *input) {

		for (int i = 0; i < nn; i++) {
			output[i] = (bias) ? 1 * weights[i][nn] : 0.0;
			for (int j = 0; j < ni; j++) {
				output[i] += input[j] * weights[i][j];
			}
		}

	}

	double* compute_output(double *input) {

		compute_potentials(input);
		
		//racunanje izlaza
		for (int i = 0; i < nn; i++)
			output[i] = activation_fn(output[i]);

		return output;

	}

	virtual matrix* derivative_by_input(double *input) = 0;		//parcijalni izvodi po ulazu
		
	virtual matrix* derivative_by_weights() = 0;			// -||- po tezinama

	void update_weights(double *new_weights) {
		weights.set(new_weights);
	}

	void print() {		//samo za testiranje (obrisati u finalnoj verziji)
		std::cout << "---------------------------------\n";
		std::cout << weights;
	}

};