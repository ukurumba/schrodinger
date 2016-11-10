# -*- coding: utf-8 -*-
import numpy as np 
import scipy.stats as ss
import scipy

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

def overall_hamiltonian(input_coefficients,potential_energy,c):
    '''This function calculates the overall hamiltonian on a wavefunction expanded as a list of basis set coefficients 
    for the Legendre polynomial basis set. 

    Input
        input_coefficients: list of floats (the wavefx expanded in the Legendre polynomial basis set)
        potential_energy: float (the potential energy constant)
        c: float (a constant that corresponds to h_bar / 2 * mass in the Hamiltonian operator)

    Output
        output_coefficients: list of floats (the basis set coefficients of the hamiltonian operator applied to the 
        input wavefunction'''

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
        mapper(fx,54,basis_set_type = 'legendre')'''
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
        for i in range(n):
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
               x is the input to the function
               i is the basis set index (>= 0)'''
            return fx(x) * fourier(i,x,wave)
        def basis_fx_integrand(x,k,wave):
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
            
        return a_cos,a_sin


def derivative_fourier(cos_in,sin_in):
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
        cos_i = [cos_in[i] * j for j in cos_i] 
        sin_i = [cos_in[i] * j for j in sin_i]
        for k in range(len(cos_i)):
            cos_out[k] += cos_i[k]

        for l in range(len(sin_i)):
            sin_out[l] += sin_i[l]

    for i in range(len(sin_in)): 
        cos_i,sin_i = hamiltonian_fourier([0] * len(cos_in),hamiltonian_table_sin[i],potential_energy,c)
        cos_i = [sin_in[i] * j for j in cos_i] 
        sin_i = [sin_in[i] * j for j in sin_i]
        for k in range(len(cos_i)):
            cos_out[k] += cos_i[k]

        for l in range(len(sin_i)):
            sin_out[l] += sin_i[l]

    output_coefficients = cos_out + sin_out
    return output_coefficients 





















