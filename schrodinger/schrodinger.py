# -*- coding: utf-8 -*-
import numpy as np 
import scipy.stats as ss

def legendre_table_generator(n):

	'''This function computes the table of legendre polynomial coefficients given a desired number of basis set functions (n).
	Input: n (integer)
	Output: Table of basis set coefficients'''

	from scipy.special import legendre
	grid = []
	for i in range(n):
	    table = []
	    poly_coeffs = legendre(i)
	    for k in range(i+1):
	        table.append(poly_coeffs[k])
	    for j in range(n-i-1):
	        table.append(0)
	    grid.append(table)
	return grid

def derivative(input_coefficients):
	'''This function takes the derivative of a Legendre polynomial. The Legendre polynomial is passed to the del function 
	as a set of coefficients to the standard polynomial basis set (1,x,x^2,x^3, etc.). The derivative is then computed on 
	this array.
	Input: list of floats (coefficients for the standard polynomial basis set)
	Output: list of floats (coefficients for the standard polynomial basis set)'''

	output_coefficients = [0] * len(input_coefficients)
	for i in range(len(input_coefficients)):
	    
	    if i != (len(input_coefficients)-1):
	        output_coefficients[i] = input_coefficients[i+1] * (i+1) #using the powers derivative rule
	    else:
	        output_coefficients[i] = 0

	return output_coefficients

def hamiltonian(input_coefficients,potential_energy,c):
	'''This function computes the hamiltonian operator on a basis set function. The basis set function is passed 
	as a set of coefficients for the standard basis set (1,x,x^2,x^3, etc.). The hamiltonian is computed by operating 
	the derivative operator on the basis set coefficients twice and adding the obtained coefficients to the potential energy 
	operator operating 	on the input basis set coefficients. The output is the basis set coefficients of the function obtained
	from the hamiltonian operator, in the basis set of the Legendre polynomials. Thus this operator takes a set of basis set 
	coefficients corresponding to the standard basis set and returns a set of basis set coefficients corresponding to the 
	Legendre polynomial basis set.

	Input: 
		input_coefficients: list of floats (input coefficients in standard basis set)
		potential_energy: float (a constant potential energy value)
		c: float (a constant that corresponds to h_bar / 2 * mass in the Hamiltonian operator)
		legendre_table: array of floats (coefficients for the legendre polynomials expanded in the standard basis set)

	Output: list of floats (output coefficients in legendre polynomial basis set)'''

	output_coefficients = derivative(derivative(input_coefficients))
	output_coefficients = [-c * i for i in output_coefficients]
	potential_energy_coefficients = [potential_energy * i for i in input_coefficients]
	for i in range(len(input_coefficients)):
		output_coefficients[i] += potential_energy_coefficients[i]

	output_coefficients = np.asarray(output_coefficients)

	#setting up matrix s.t. each column is a legendre polynomial expanded in the standard basis
	legendre_table = legendre_table_generator(len(input_coefficients))
	legendre_table = (np.asarray(legendre_table)).transpose() 
	legendre_table = np.linalg.inv(legendre_table) #creating correct change-of-basis matrix													  

	return np.dot(legendre_table,output_coefficients) #outputs in order [P_0(x),P_1(x),P_2(x),etc.]








