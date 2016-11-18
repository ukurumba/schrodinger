# -*- coding: utf-8 -*-
import numpy as np 
import scipy.stats as ss
import scipy.integrate as integrate
import scipy
from scipy.special import legendre
from scipy.special import eval_legendre

def legendre_table_generator(n):

    '''This function computes the table of legendre polynomial coefficients given a desired number of basis set functions (n).
    The polynomials are outputted in descending order and read from lowest to highest degree. For example, 
    2x^3 + x2 is encoded as [0,0,1,2]. 
    The 
    Input
        n: integer (number of basis set functions)
    Output
        grid: array of floats  (table of basis set coefficients)'''

    from scipy.special import legendre
    grid = []
    for i in range(n):
        table = []
        poly_coeffs = legendre(i) #gathering polynomial coefficients
        for k in range(i+1):
            table.append(poly_coeffs[k])
        for j in range(n-i-1):
            table.append(0)
        grid.append(table)
    return grid

def overall_hamiltonian_legendre(input_coefficients,potential_energy,c):
    '''This function calculates the overall hamiltonian on a wavefunction expanded as a list of basis set coefficients 
    for the Legendre polynomial basis set. 

    Input
        input_coefficients: list of floats (the wavefx expanded in the Legendre polynomial basis set)
        potential_energy: float (the potential energy constant)
        c: float (a constant that corresponds to h_bar / 2 * mass in the Hamiltonian operator)

    Output
        output_coefficients: list of floats (the basis set coefficients of the hamiltonian operator applied to the 
        input wavefunction expanded in the Legendre basis set)'''

    legendre_table = legendre_table_generator(len(input_coefficients))
    output_coefficients = [0 + 0j] * len(input_coefficients)
    for i in range(len(input_coefficients)):
        output_i = hamiltonian(legendre_table[i],potential_energy,c)
        output_i = [input_coefficients[i] * j for j in output_i] 
        # ^ multipling entire representation of hamiltonian applied to basis set function by the input basis 
        #   set coefficient
        for k in range(len(input_coefficients)):
            output_coefficients[k] += output_i[k] #add basis set coefficients

    return output_coefficients 

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
            output_coefficients[i] = 0 #manually setting coefficient of max degree of input as 0. 

    return output_coefficients

def hamiltonian(input_coefficients,potential_energy,c):
    '''This function computes the hamiltonian operator on a basis set function. The basis set function is passed 
    as a set of coefficients for the standard basis set (1,x,x^2,x^3, etc.). The hamiltonian is computed by operating 
    the derivative operator on the basis set coefficients twice and adding the obtained coefficients to the potential energy 
    operator operating  on the input basis set coefficients. The output is the basis set coefficients of the function obtained
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

def mapper(fx,n,basis_set_type):
    '''Returns a mapping of an input function to the selected basis set (Legendre Polynomials or Fourier Series).

    Input
        fx: function (the name of a function of one variable, can produce complex output but must take real input)
        n: integer (the number of desired basis set functions. n must be odd for the fourier series basis set)
        basis_set_type: 'legendre' or 'fourier' (the type of function in the basis set)

    Output
        output_coefficients: array of complex numbers (the function represented as coefficients for the chosen basis set)

    Example
        fx = x +2 - x**2 
        output = schrodinger.mapper(fx,54,basis_set_type = 'legendre')
        cos_out, sin_out = schrodinger.mapper(fx,55,basis_set_type = 'fourier'''

    import scipy.integrate as integrate
    if basis_set_type == 'legendre':
        from scipy.special import eval_legendre
        
        import scipy 
        
        def inner_product_integrand(x,fx,i): 
            '''fx is the input function
               x is the input to the function
               i is the Legendre Polynomial number (>= 0)'''
            return fx(x) * eval_legendre(i,x)

        a = []
        for i in range(n): #split real and imaginary components of integrand (saw on StackExchange) so quad can handle them 
            inner_product_real, error_inner_product_real = integrate.quad(lambda x: scipy.real(inner_product_integrand(x,fx,i)),-1,1)
            inner_product_imag, error_inner_product_imag = integrate.quad(lambda x: scipy.imag(inner_product_integrand(x,fx,i)),-1,1)
            inner_product = inner_product_real + inner_product_imag *1j
            normalizing_coefficient, error_normalizing_coefficient = integrate.quad(lambda i,x: eval_legendre(x,i)**2, -1,1,args=(i))
            a.append(inner_product/normalizing_coefficient)
        return a

    elif basis_set_type == 'fourier':
        if int(n/2) == n/2:
            raise ValueError('For a fourier basis set choice, the number of basis set functions must be odd')
        num_cos = int((n+1) / 2)
        num_sin = int((n-1) / 2)
    
        a_cos = []
        a_sin = []

        def fourier(k,x,wave): 
            if wave == 'sine':
                return np.sin(k * x)
            elif wave =='cosine':
                return np.cos(k*x)
        def inner_product_integrand(x,fx,i,wave):
            '''fx is the input function
               x is the input toset the function
               i is the basis  index (>= 0)'''
            return fx(x) * fourier(i,x,wave)
        def basis_fx_integrand(x,k,wave): #normalizing coefficient integrand
            if wave == 'sine':
                return (np.sin(k*x))**2
            elif wave == 'cosine':
                return (np.cos(k*x))**2
            
        import scipy            
        for i in range(num_cos):
            inner_product_real, error_inner_product_real = integrate.quad(lambda x: scipy.real(inner_product_integrand(x,fx,i,'cosine')),-np.pi,np.pi)
            inner_product_imag, error_inner_product_imag = integrate.quad(lambda x: scipy.imag(inner_product_integrand(x,fx,i,'cosine')),-np.pi,np.pi)
            inner_product = inner_product_real + inner_product_imag *1j
            normalizing_coefficient, error_normalizing_coefficient = integrate.quad(lambda x: basis_fx_integrand(x,i,'cosine'),-np.pi,np.pi)
            a_cos.append(inner_product/normalizing_coefficient)
            
        for i in range(1,num_sin+1,1):
            inner_product_real, error_inner_product_real = integrate.quad(lambda x: scipy.real(inner_product_integrand(x,fx,i,'sine')),-np.pi,np.pi)
            inner_product_imag, error_inner_product_imag = integrate.quad(lambda x: scipy.imag(inner_product_integrand(x,fx,i,'sine')),-np.pi,np.pi)
            inner_product = inner_product_real + inner_product_imag *1j
            normalizing_coefficient, error_normalizing_coefficient = integrate.quad(lambda x: basis_fx_integrand(x,i,'sine'),-np.pi,np.pi)
            a_sin.append(inner_product/normalizing_coefficient)
            
        return a_cos,a_sin #keeping cosine and sine coefficients separate to facilitate taking the derivative

def derivative_fourier(cos_in,sin_in):
    '''This function takes the derivative of a function passed in as a set of cosine and sine basis set coefficients. Internal function.'''
    sin_out = [-i*cos_in[i] for i in range(1,len(sin_in)+1,1)]
    cos_out = [0] + [(i+1) * sin_in[i] for i in range(len(sin_in))]
    return cos_out,sin_out

def hamiltonian_fourier(cos_in,sin_in,potential_energy,c):
    '''This function computes the hamiltonian on a function represented by 2 lists of basis set coefficients,
    one for the cosines and one for the sines.

    Input
        cos_in: list of complex numbers (input coefficients for cosine fxs indexed at k=0)
        sin_in: list of complex numbers (input coefficients for sine fxs indexed at k=1)
        potential_energy: float (constant that represents potential energy)
        c: float(constant that is part of hamiltonian operator: h_bar / 2 * maass)

    Output
        cos_out, sin_out: list of complex numbers, list of complex numbers (coefficients with same indexing and respective fxs)
    '''

    cos_1,sin_1 = derivative_fourier(cos_in,sin_in)
    cos_2, sin_2 = derivative_fourier(cos_1,sin_1)

    cos_out = [-c * i for i in cos_2]
    sin_out = [-c * i for i in sin_2]
    potential_energy_coefficients_cos = [potential_energy * i for i in cos_in]
    potential_energy_coefficients_sin = [potential_energy * i for i in sin_in]

    for i in range(len(cos_in)):
        cos_out[i] += potential_energy_coefficients_cos[i]

    for i in range(len(sin_in)):
        sin_out[i] += potential_energy_coefficients_sin[i]

    return cos_out, sin_out 

def overall_hamiltonian_fourier(cos_in,sin_in,potential_energy,c):
    '''Returns the coefficients of the output of the hamiltonian operator applied to the fourier expansion of an input function.
    Input
        cos_in: list of complex numbers (input coefficients for the cosine basis set fxs)
        sin_in: list of complex numbers (input coefficients for the sine basis set fxs)
        potential energy: float (the potential energy)
        c: float (the constant in the hamiltonian operator corresponding to h_bar ^2 / 2 * mass)
    Output
        output_coefficients: list of complex numbers (output coefficients for both the cosine and sine basis set fxs)'''

    cos_out = [0 + 0j] * len(cos_in)
    sin_out = [0 + 0j] * len(sin_in)
    hamiltonian_table_cos = np.zeros((len(cos_in),len(cos_in)))
    for i in range(len(hamiltonian_table_cos)):
        hamiltonian_table_cos[i,i] = 1
    hamiltonian_table_sin = np.zeros((len(sin_in),len(sin_in)))
    for i in range(len(hamiltonian_table_sin)):
        hamiltonian_table_sin[i,i] = 1

    for i in range(len(cos_in)): 
        cos_i,sin_i = hamiltonian_fourier(hamiltonian_table_cos[i],[0]*len(sin_in),potential_energy,c)
        cos_i = [cos_in[i] * j for j in cos_i] #multiplying entire cos_i by the input coefficient for that basis function representation
        sin_i = [cos_in[i] * j for j in sin_i] #multiplying entire sin_i by the input coefficient for that basis function representation
        for k in range(len(cos_i)):
            cos_out[k] += cos_i[k]

        for l in range(len(sin_i)):
            sin_out[l] += sin_i[l]

    for i in range(len(sin_in)): #repeating calculation for sine functions
        cos_i,sin_i = hamiltonian_fourier([0] * len(cos_in),hamiltonian_table_sin[i],potential_energy,c)
        cos_i = [sin_in[i] * j for j in cos_i] 
        sin_i = [sin_in[i] * j for j in sin_i]
        for k in range(len(cos_i)):
            cos_out[k] += cos_i[k]

        for l in range(len(sin_i)):
            sin_out[l] += sin_i[l]

    output_coefficients = cos_out + sin_out
    return output_coefficients 

def coeffs_to_fx(input_coefficients,x,coeff_type):
    '''This function evaluates a polynomial function (represented on the Legendre or standard basis set) at a point x. The input coefficients correspond 
    to the polynomial basis set values 
    Input
        input_coefficients: 1D array of complex numbers (input coefficients in either Legendre or standard basis set)
        x: float (the x value at which the function is to be evaluated)
        coeff_type: string, choose from: 'legendre' or 'standard' (what basis set the input coefficients represent)
    Output
        value: float (output value)'''

    value = 0
    if coeff_type == 'legendre':
        for i in range(len(input_coefficients)): 
            value += input_coefficients[i] * eval_legendre(i,x) #eval_legendre returns value of ith legendre polynomial at pt x
    elif coeff_type == 'standard':
        for i in range(len(input_coefficients)):
            value += input_coefficients[i] * x ** i

    return value

def hamiltonian_element(coefficients_i,coefficients_j,potential_energy,c):
    '''This function computes the i,jth element of the Hamiltonian matrix. It does so by numerically integrating the product 
    of the complex conjugate of the ith basis function and the hamiltonian operating on the jth basis function.
    Input
        coefficients_i: list of complex floats (the basis set coefficients of vector i in the standard basis set [1,x,x^2,x^3,etc.])
        coefficients_j: list of complex floats (the basis set coefficients of vector j in the standard basis set)
        potential_energy: float (the potential energy constant)
        c: float (the constant involved in the Hamiltonian operator)
        
    Output
        matrix_element: float (the value of the above numerical integral)'''
    
    
    import scipy.integrate as integrate
    hamiltonian_on_phi_j = hamiltonian(coefficients_j,potential_energy,c) #operating hamiltonian on function j, returns in legendre basis set
    def value(x): #creating a fx to pass to scipy.integrate.quad
        phi_i = np.conjugate(coeffs_to_fx(coefficients_i,x, coeff_type = 'standard'))
        phi_j = coeffs_to_fx(hamiltonian_on_phi_j,x,coeff_type = 'legendre')
        return phi_i * phi_j
    matrix_element_real, error_real = integrate.quad(lambda x: scipy.real(value(x)),-1,1)
    matrix_element_imaginary,error_imaginary = integrate.quad(lambda x: scipy.imag(value(x)),-1,1)
    matrix_element = matrix_element_real + matrix_element_imaginary* 1j
    
    return matrix_element
        
def hamiltonian_matrix(n,potential_energy,c):
    '''This computes a matrix where each component is <Phi_i | H | Phi_j> where H is the hamiltonian operator.'''
    legendre_table = legendre_table_generator(n)
    hamil_matrix = np.zeros((len(legendre_table),len(legendre_table)))
    for i in range(len(legendre_table)):
        for j in range(len(legendre_table)):
            hamil_element = hamiltonian_element(legendre_table[i],legendre_table[j],potential_energy,c) #legendre(i) returns legendre coefficients
            hamil_matrix[i,j] = scipy.real(hamil_element)
    return hamil_matrix

def ground_state_wavefx(n,potential_energy,c,basis_set_type = 'legendre', domain = (-1,1)):
    '''The main function in this program. Computes the ground state wavefunction for a given potential energy and mass (represented as
    the constant c in the hamiltonian operator). Does so by solving for the eigenvalues (energies) and eigenvectors (wavefxs) of the hamiltonian matrix.
    Input
        n: positive integer (the number of basis set functions desired)
        potential_energy: float (the constant potential energy value)
        c: float(the constant in the hamiltonian operator equal to h_bar ^ 2 / 2 * mass)
        basis_set_type: string (optional) choose from: 'legendre' or 'fourier' (the type of basis set desired. Defaults to legendre)
        domain: tuple (optional) (the domain on which the fourier basis set is to be evaluated. Defaults to (-1,1)) Note: Domain 
        cannot be altered for the legendre basis set (will produce an error)

    Output
        (eval, evec) : float, array of floats (eval = the minimum energy of the system. evec = best approximation of the groundstate
        wavefx, represented in the chosen basis set) Note: for fourier basis set, outputs basis set functions in the order 
        (1, cos(x),cos(2x),...,sin(x),sin(2x), ...)

    Example
        n = 11
        potential_energy = 15
        c = 16
        wavefx = schrodinger.ground_state_wavefx(n,potential_energy,c,basis_set_type = 'fourier',domain = (-2,2))'''




    if basis_set_type == 'legendre':
        hamil_matrix = hamiltonian_matrix(n,potential_energy,c)
        evals,evecs = np.linalg.eigh(hamil_matrix)
        return min(evals), evecs[np.argmin(evals)]
    elif basis_set_type == 'fourier':
        hamil_matrix = hamiltonian_matrix_fourier(n,potential_energy,c,domain)
        evals,evecs = np.linalg.eigh(hamil_matrix)
        return min(evals), evecs[np.argmin(evals)]

def coeffs_to_fx_fourier(cos_in,sin_in,x):
    '''This function takes a function represented on the fourier basis set and evaluates it at a point x.'''
    value = 0
    for i in range(len(cos_in)):
        value += cos_in[i] * np.cos(i * x)
    for i in range(len(sin_in)):
        value += sin_in[i] * np.sin((i+1) * x)
    return value 

def hamiltonian_element_fourier(cos_in_i,sin_in_i,cos_in_j,sin_in_j,potential_energy,c,domain):
    '''This function computes an element of the hamiltonian matrix, doing so by numerically computing the integral of 
    the product of the conjugate of phi i and the hamiltonian operated on phi_j.'''

    def value(x):
        phi_i = np.conjugate(coeffs_to_fx_fourier(cos_in_i,sin_in_i,x))
        cos_j,sin_j = hamiltonian_fourier(cos_in_j,sin_in_j,potential_energy,c)
        phi_j = coeffs_to_fx_fourier(cos_j,sin_j,x)
        return phi_i * phi_j
    
    matrix_element_real, error_real = integrate.quad(lambda x: scipy.real(value(x)),domain[0],domain[1])
    matrix_element_imaginary, error_imaginary = integrate.quad(lambda x: scipy.imag(value(x)),domain[0],domain[1])
    matrix_element = matrix_element_real + matrix_element_imaginary * 1j
    return matrix_element 

def hamiltonian_matrix_fourier(n,potential_energy,c,domain = (-1,1)):
    '''Evaluates the hamiltonian matrix in the fourier basis set, where each element i,j of the matrix is <phi_i | H | phi_j>
    where H is the hamiltonian operator with the given potential energy and c value. i and j correspond to selections from the 
    list (1,cos(x),cos(2x),...,sin(x),sin(2x),...) etc.'''
    num_cos = int((n+1)/2)
    num_sin = int((n-1)/2)
    cos_table = np.zeros((num_cos,num_cos))
    sin_table = np.zeros((num_sin,num_sin))
    for i in range(len(cos_table)):
        cos_table[i,i] = 1
    for i in range(len(sin_table)):
        sin_table[i,i] = 1
    hamil_matrix = np.zeros((n,n))
    for i in range(n): #this is not the best way to do it but since I've already used this division of cos/sin ... 
        if i < num_cos:
            cos_in_i = cos_table[i]
            sin_in_i = [0] * num_sin
        elif i >= num_cos:
            cos_in_i = [0] * num_cos
            sin_in_i = sin_table[i - num_cos]
        for j in range(n):
            if j < num_cos:
                cos_in_j = cos_table[j]
                sin_in_j = [0] * num_sin
            elif j >= num_cos:
                cos_in_j = [0] * num_cos
                sin_in_j = sin_table[j-num_cos]
            hamil_matrix[i,j] = scipy.real(hamiltonian_element_fourier(cos_in_i,sin_in_i,cos_in_j,sin_in_j,potential_energy,c,domain))      
    return hamil_matrix
        
    






















