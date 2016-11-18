=====
Usage
=====

To use schrodinger in a project::

    import schrodinger

To calculate the lowest-energy wavefunction for a given potential energy and c constant::
	help(schrodinger.ground_state_wavefx)
	>>> The main function in this program. Computes the ground state wavefunction for a given potential energy and mass (represented as
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
        wavefx = schrodinger.ground_state_wavefx(n,potential_energy,c,basis_set_type = 'fourier',domain = (-2,2))

To evaluate the hamiltonian on an input function requires using two functions (note that while the mapper function is the same 
for either the legendre or fourier basis set choice, the actual hamiltonian evaluation function is different. See below::

	def fx(x):
		return x**5

	number_of_basis_set_fxs = 57
	potential_energy = 12
	c = 15

	input_coefficients_legendre = schrodinger.mapper(fx,number_of_basis_set_fxs,'legendre')
	input_cosine_coefficients, input_sine_coefficients = schrodinger.mapper(fx,number_of_basis_set_fxs,'fourier')

	legendre_hamiltonian = schrodinger.overall_hamiltonian_legendre(input_coefficients_legendre, potential_energy,c)
	fourier_hamiltonian = schrodinger.overall_hamiltonian_fourier(input_cosine_coefficients, input_sine_coefficients, potential_energy,c)

	help(schrodinger.mapper)
	>>> Returns a mapping of an input function to the selected basis set (Legendre Polynomials or Fourier Series).

    Input
        fx: function (the name of a function of one variable, can produce complex output but must take real input)
        n: integer (the number of desired basis set functions. n must be odd for the fourier series basis set)
        basis_set_type: 'legendre' or 'fourier' (the type of function in the basis set)

    Output
        output_coefficients: array of complex numbers (the function represented as coefficients for the chosen basis set)

    Example
        fx = x +2 - x**2 
        output = schrodinger.mapper(fx,54,basis_set_type = 'legendre')
        cos_out, sin_out = schrodinger.mapper(fx,55,basis_set_type = 'fourier'

    help(schrodinger.overall_hamiltonian_legendre)
    >>> This function calculates the overall hamiltonian on a wavefunction expanded as a list of basis set coefficients 
    for the Legendre polynomial basis set. 

    Input
        input_coefficients: list of floats (the wavefx expanded in the Legendre polynomial basis set)
        potential_energy: float (the potential energy constant)
        c: float (a constant that corresponds to h_bar / 2 * mass in the Hamiltonian operator)

    Output
        output_coefficients: list of floats (the basis set coefficients of the hamiltonian operator applied to the 
        input wavefunction expanded in the Legendre basis set)

    help(schrodinger.overall_hamiltonian_fourier)
    >>> Returns the coefficients of the output of the hamiltonian operator applied to the fourier expansion of an input function.
    Input
        cos_in: list of complex numbers (input coefficients for the cosine basis set fxs)
        sin_in: list of complex numbers (input coefficients for the sine basis set fxs)
        potential energy: float (the potential energy)
        c: float (the constant in the hamiltonian operator corresponding to h_bar ^2 / 2 * mass)
    Output
        output_coefficients: list of complex numbers (output coefficients for both the cosine and sine basis set fxs)
        



