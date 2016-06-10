#pragma once
#include<iostream>
#include <random>

#include "linear_layer.h"
#include "tanhip_layer.h"
#include "sig_layer.h"

const double min_ran = -0.1;
const double max_ran = 0.1;
int n_layers = 2; //ovde jer ih koristi klasa layer (extern)
layer** layers; //za backprop, trenutni sloj mora da zna tip sledeceg
//const double learning_rate = ?;
//const double momentum = ?;

template<typename H, typename O>
class neural_network {
protected:
	int tin, thl, tol; //pamtio ove 3 vrednosti samo zbog print f-ja, testiranje
	double* weights;		//tezine za celu mrezu
	double* output;			//rezultat mreze, moze da bude niz, kod nas jedan double jer imamo jedan out neuron
	double* deltas;			//delte za sve neurone
public:
	neural_network(int in_num, int hl_num, int ol_num = 1) {
		tin = in_num;				
		thl = hl_num;
		tol = ol_num;
		output = nullptr;
		initialize_weights((in_num + 1) * hl_num + (hl_num + 1) * ol_num);
		initialize_deltas(hl_num + ol_num);
		layers = new layer*[2];
		layers[0] = new H(hl_num, in_num, 1, weights, deltas, 0);
		layers[1] = new O(ol_num, hl_num, 1, weights + (in_num + 1) * hl_num, deltas + hl_num, 1);
	}

	virtual ~neural_network() {
		delete layers[0];
		delete layers[1];
		delete[] layers;
		delete[] weights;
		delete[] deltas;
	}

	void initialize_weights(int weights_size) {
		weights = new double[weights_size];
		std::random_device device;
		std::default_random_engine engine(device());
		std::uniform_real_distribution<double> distribution(min_ran, max_ran);
		for (int i = 0; i < weights_size; i++) {
			weights[i] = distribution(engine);
		}
	}

	void initialize_deltas(int deltas_size) {
		deltas = new double[deltas_size];
	}
	
	virtual double* compute_output(double *input) {     //da bi mogla da se predefinise
		output = layers[1]->compute_output(layers[0]->compute_output(input));
		return output;
	}

	void backPropagate(double p = 1.0) { //p je ocekivani izlaz za dati ulaz
		layers[1]->compute_deltas(p);
		layers[1]->update_weights();
		layers[0]->update_weights();
	}

	double* get_output() {
		return output;
	}

	void printDeltas() {
		for (int i = 0; i < thl + tol; i++)
			std::cout << deltas[i] <<  " ";
		std::cout << "deltas" << std::endl << std::endl;
	}

	void printWeights() {
		for (int i = 0; i < thl; i++) {
			for (int j = 0; j < tin + 1; j++)
				std::cout << weights[i*thl + j] << " ";
			std::cout << std::endl;
		}
		std::cout << std::endl;
		for (int i = (tin + 1) * thl; i < (tin + 1) * thl + (thl + 1) * tol; i++)
			std::cout << weights[i] << " ";
		std::cout << std::endl;
	}
};