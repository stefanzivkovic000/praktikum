#pragma once

#include <iostream>

class matrix;

class operation {
public:
	const matrix &a;
	const matrix &b;
	operation(const matrix &a, const matrix &b) : a(a), b(b) { }
};

//********** SUMMATION **********

class summation: public operation {
public:
	summation(const matrix &a, const matrix &b) : operation(a, b) { }
};

const summation operator+(const matrix &a, const matrix &b) {
	return summation(a, b);
}

//********** SUBTRACTION **********

class subtraction: public operation {
public:
	subtraction(const matrix &a, const matrix &b) : operation(a, b) { }
};



const subtraction operator-(const matrix &a, const matrix &b) {
	return subtraction(a, b);
}

//********** MULTIPLICATION **********

class multiplication: public operation {
public:
	multiplication(const matrix &a, const matrix &b) : operation(a, b) { }
};

const multiplication operator*(const matrix &a, const matrix &b) {
	return multiplication(a, b);
}