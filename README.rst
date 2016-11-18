===============================
schrodinger
===============================


.. image:: https://img.shields.io/travis/ukurumba/schrodinger.svg
        :target: https://travis-ci.org/ukurumba/schrodinger

.. image:: https://codecov.io/gh/ukurumba/schrodinger/branch/develop/graph/badge.svg
		:target: https://codecov.io/gh/ukurumba/schrodinger 


A Python application that does one of two things:

* Given a potential energy and other constants, returns the lowest energy groundstate wavefunction and minimal energy value 
* Evaluates the Hamiltonian operator on a given function 

For the first application, the program works by setting up a basis set of a user-decided size in the user-decided type (Fourier or Legendre) and then computing the lowest-energy wavefunction on this basis set by solving the hamiltonian matrix for the lowest energy eigenvalue/eigenvector. The program works by mapping the given function to a basis set of choice (Fourier or Legendre) and computing the Hamiltonian on this basis set. 

Examples of usage:

-Calculating wavefunction.
:: 
	number_of_basis_set_fxs = 12
	potential_energy = 15
	c = 12
	energy, wavefunction = schrodinger.ground_state_wavefx(number_of_basis_set_fxs,potential_energy,c,basis_set_type = 'fourier',domain = (-1,1))

-Operating Hamiltonian operator on a given wavefunction.
:: 
	def fx(x):
		return x**5

	number_of_basis_set_fxs = 57
	potential_energy = 12
	c = 15

	input_coefficients_legendre = schrodinger.mapper(fx,number_of_basis_set_fxs,'legendre')
	input_cosine_coefficients, input_sine_coefficients = schrodinger.mapper(fx,number_of_basis_set_fxs,'fourier')

	legendre_hamiltonian = schrodinger.overall_hamiltonian_legendre(input_coefficients_legendre, potential_energy,c)
	fourier_hamiltonian = schrodinger.overall_hamiltonian_fourier(input_cosine_coefficients,input_sine_coefficients,potential_energy,c)

See Usage for more specific documentation. 






* Free software: MIT license
* Documentation: https://github.com/ukurumba/schrodinger/blob/master/docs/usage.rst



Credits
---------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage

