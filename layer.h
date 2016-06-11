#pragma once

#include <iostream>
const double eta = 0.9;

class layer {
protected:
	layer* next_layer;	//desni sloj, kao ga nema, onda je ovo output sloj
	int nn;			//broj neurona u sloju
	int ni;			//broj ulaza
	bool bias;		//1 ako ima bias, 0 inace. ovaj bias je prakticno neuron u prethodnom sloju, imaginaran je								
	double* weights;  //pokazivac na pocetak njegovih tezina u globalnom nizu weights
	double* potentials;   
	double* output;
	double* inputs; //dodao zbog backProp, treba za jedan parcijalni izvod
	double* deltas; //slicno kao za weights
	virtual double activation_fn(double a) = 0;			//a je potencijal
	virtual double activation_fn_prime(double a) = 0;	
	/*double cost_fn(double x, double y) { //mse, ne koristi se ova fja
		return (y-x)*(y-x);
	}*/ 
	double cost_fn_prime(double x, double y) {
		return -2.0 * (y-x);
	}
	
public:
	layer(int nn, int ni, bool bias, double* weights, double* deltas, layer* next_layer=nullptr) {
		this -> nn = nn;
		this -> ni = ni;
		this -> bias = bias;
		this -> weights = weights;
		this -> deltas = deltas;
		this -> output = new double[nn];
		this -> next_layer = next_layer;
		potentials = new double[nn]();
	}

	~layer() {
		delete[] potentials;
		delete[] inputs;
		delete[] output;
	}

	virtual double d_ai_xj(int i, int j) = 0; // za linearni sloj dovoljni argumenti
	virtual double d_ai_wij(int j) = 0;

	void compute_potentials(double *input) {		
		inputs = new double[ni]; //bias je imaginaran :)
		for (int i = 0; i < ni; i++)
			inputs[i] = input[i]; 

		for (int i = 0; i < nn; i++) {
			potentials[i] = (bias) ? 1 * weights[i * (ni + 1)] : 0.0;
			for (int j = 0; j < ni; j++) {
				potentials[i] += input[j] * weights[i * (ni + bias) + j + bias];
			}
		}

	}

	double* compute_output(double *input) {
		
		compute_potentials(input);

		for (int i = 0; i < nn; i++)
			output[i] = activation_fn(potentials[i]);

		return output;

	}

	void update_weights(double *new_weights) { //treba za Kalmana
		int to = (ni + 1) * nn;
		for (int i = 0; i < to; i++) 
			weights[i] = new_weights[i];
	}

	void update_weights() {
		for (int i = 0; i < nn; i++) {
			weights[i*(ni + bias)] -= (bias) ? eta * deltas[i] * 1 : 0.0; // update za bias ide ovako
			for (int j = bias; j < ni + bias; j++)
				weights[i*(ni + bias) + j] -= eta * deltas[i] * this->d_ai_wij(j-bias);
		} 
	}

    void compute_deltas(double y=1.0) { //cisto onako =1, da poziv za skriveni sloj ne bi imao args
		if (!next_layer) {
			deltas[0] = cost_fn_prime(output[0],y)*this->activation_fn_prime(potentials[0]);
		}
		else {
			for (int i = 0; i < nn; i++) {
				deltas[i] = deltas[nn] * next_layer->d_ai_xj(0,i+1)*this->activation_fn_prime(potentials[i]); //
			} // drugacije bi bilo da ima vise od 2 sloja
		}
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