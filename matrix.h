#pragma once

#include <iostream>

#include "matrix_operations.h"

class matrix {						//podaci matrice se cuvaju u jednodimenzionalnom nizu
public:
	int m;
	int n;
	double *data;

	matrix() {
		m = 0;
		n = 0;
		data = nullptr;
	}
	
	matrix(int m, int n) {				//konstruktor koji rezervise matricu sa trash vrednostima
		this -> m = m;
		this -> n = n;
		data = new double[n * m];
	}

	matrix(int m, int n, double *data) {		//konstruktor koji dobija popunjenu matricu
		this -> m = m;
		this -> n = n;
		this -> data = new double[m * n];
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) 
				(this -> data)[i * n + j] = data[i * n + j];
		}
	}

	matrix(const matrix &mat) {			//konstruktor kopije
		operator=(mat);
	}

	matrix(const summation &sum) {
		operator=(sum);
	}

	~matrix() {
		delete [] data;
	}

	double* get_pointer_to_data() {
		return data;
	}

	void resize(int m, int n) { 
		delete [] data;
		this -> m = m;
		this -> n = n;
		data = new double[m * n];
	}

	void set(double *values, int size = 0) {
		if (!size)
			size = m * n;
		for (int i = 0; i < size; i++) {
			data[i] = values[i];
		}
	}

	double& element(int i, int j) {									//podrazumeva se da prvi red i prva kolona imaju indeks 0
		if (i < 0 || i >= m || j < 0 || j >= n) {						//tj. prvom elementu prve kolone je na poziciji 0,0
			std::cout << "Neispravan indeks dohvatanja elementa\n";
		}
		return data[i * n + j];
	}

	double* operator[](int i) {
		if (i < 0 || i >= m) {
			std::cout << "Neispravan indeks dohvatanja elementa\n";
		}
		return data + i*n;
	}

	const matrix& operator=(const matrix &mat) {
		delete [] data;
		m = mat.m;
		n = mat.n;
		data = new double[m * n];
		for (int i = 0; i < m * n; i++)
			data[i] = mat.data[i];
		return *this;
	}

	const matrix& operator=(const summation &sum) {
		if (sum.a.n != sum.b.n || sum.a.m != sum.b.m) {
			std::cout << "Neispravna dimenzija cinioca sabiranja\n";
		}
		for (int i = 0; i < sum.a.m * sum.a.n; i++)
			data[i] = sum.a.data[i] + sum.b.data[i];
		return *this;
	}

	const matrix& operator=(const subtraction &sub) {
		if (sub.a.n != sub.b.n || sub.a.m != sub.b.m) {
			std::cout << "Neispravna dimenzija cinioca oduzimanja\n";
		}
		for (int i = 0; i < sub.a.m * sub.a.n; i++)
			data[i] = sub.a.data[i] - sub.b.data[i];
		return *this;
	}

	const matrix& operator=(const multiplication &mul) {
		if (mul.a.n != mul.b.m) {
			std::cout << "Neispravna dimenzija cinioca mnozenja\n";
		}
		delete [] data;
		m = mul.a.m;
		n = mul.b.n;
		data = new double[m * n];
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				data[i * n + j] = 0;
				for (int k = 0; k < mul.a.n; k++) {
					data[i * n + j] += mul.a.data[i * n + k] * mul.b.data[k * n + j];
				}
			}
		}
		return *this;
	}

	const matrix& trans() {
		double *tmp = new double[m * n];
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				tmp[j * m + i] = data[i * n + j];
			}
		}
		delete [] data;
		data = tmp;
		int tmp_m = m;
		m = n;
		n = tmp_m;
		return *this;
	}



	friend std::ostream& operator<<(std::ostream &o, const matrix &mat);				//sluzi samo za testiranje, moze da se izbrise definicija

};

std::ostream& operator<<(std::ostream &o, const matrix& mat) {
	int n = mat.m * mat.n;
	for (int i = 0; i < n; i++) {
		std::cout << mat.data[i] << "\t";
		if ((i + 1) % mat.n == 0)
			std::cout << "\n";
	}
	return o;
}
