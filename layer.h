#pragma once

#include <iostream>
class layer;
extern int nLayers;
extern layer** layers;
const double eta = 0.8;

class layer {
protected:
	int my_index;		//indeks sloja u mrezi - skriveni sloj - 0, izlazni sloj - 1
	int nn;			//broj neurona u sloju
	int ni;			//broj ulaza
	bool bias;		//1 ako ima bias, 0 inace								
	double* weights;
	double* potentials;
	double* output;
	double* inputs; //dodao zbog backProp
	double* deltas;
	virtual double activation_fn(double a) = 0;			//a je potencijal
	virtual double activation_fn_inverse(double x) = 0;
	virtual double activation_fn_prime(double a) = 0;		//a je potencijal
	double cost_fn(double x, double y) { //mse
		return (y-x)*(y-x);
	}
	double cost_fn_prime(double x, double y) {
		return -2.0 * (y-x);
	}
	
public:
	virtual double d_ai_xj(int i, int j, double x) = 0;
	virtual double d_ai_wij(int j) = 0; // za linearni sloj dovoljno
	layer(int nn, int ni, bool bias, double* weights, double* deltas, int my_index) {
		
		this -> nn = nn;
		this -> ni = ni;
		this -> bias = bias;
		this -> weights = weights;
		this -> deltas = deltas;
		this -> output = new double[nn];
		this -> my_index = my_index;

		potentials = new double[nn];

	}
	~layer() {
		delete [] potentials;
		delete[] inputs;
		delete [] output;
	}
	void compute_potentials(double *input) {		
		inputs = new double[ni + bias];
		for (int i = 0; i < ni + bias; i++)
			inputs[i] = input[i]; //kopiram jer ne znam sto ne radi xD


		for (int i = 0; i < nn; i++) {
			potentials[i] = (bias) ? 1 * weights[i * (ni + bias)] : 0.0;
			for (int j = 0; j < ni; j++) {
				potentials[i] += input[j] * weights[i * (ni + bias) + j + 1];
			}
		}

	}

	double* compute_output(double *input) {
		
		compute_potentials(input);

		for (int i = 0; i < nn; i++)
			output[i] = activation_fn(potentials[i]);

		return output;

	}

	/*void update_weights(double *new_weights) {
		int to = (ni + 1) * nn;
		for (int i = 0; i < to; i++) 
			weights[i] = new_weights[i];
	}*/
	void update_weights() {
		for (int i = 0; i < nn; i++) {
			weights[i*(ni + bias)] -= eta * deltas[i] * 1; // delta za bias ide ovako
			for (int j = 1; j < ni + bias; j++)
				weights[i*(ni + bias) + j] -= eta * deltas[i] * this->d_ai_wij(j);
		}
	}

	int get_nn() {
		return nn;
	}
    void compute_deltas(double y=1) { //cisto onako =1, da poziv za skriveni sloj ne bi imao args
		if (my_index == nLayers - 1) {
			deltas[0] = cost_fn_prime(output[0],y)*this->activation_fn_prime(potentials[0]);
		}
		else {
			if (bias)
				deltas[0] = deltas[nn + 1]*1.0*this->activation_fn_prime(potentials[0]); // bias se updateuje ovako
			for (int i = 0 + bias; i < nn + bias; i++) {
				deltas[i] = deltas[nn + bias] * layers[my_index+1]->d_ai_xj(0,i,output[i])*this->activation_fn_prime(potentials[i]); //
			} // drugacija bi petlja bila da ima vise od 2 sloja (imali bi 2)
		}
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