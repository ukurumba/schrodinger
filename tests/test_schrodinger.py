#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_schrodinger
----------------------------------

Tests for `schrodinger` module.
"""


import sys
import unittest

from schrodinger import schrodinger



import numpy as np
class TestSchrodinger(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_000_something(self):
        pass

    def test_legendre_table_generator(self):
        table = schrodinger.legendre_table_generator(3)
        test_table = [[1,0,0],[0,1,0],[-0.5,0,1.5]]
        self.assertEqual(table,test_table)

    def test_derivative(self):
        table = schrodinger.legendre_table_generator(5)
        self.assertEqual(schrodinger.derivative(schrodinger.derivative(table[3])),[0,15,0,0,0])

    def test_hamiltonian(self):
        for i in range(6):
            self.assertEqual(schrodinger.hamiltonian([0,0,.25,0,-.125,0],0,1)[i],[0,0,1,0,0,0][i])

    def test_overall_hamiltonian(self):
        input_coefficients = schrodinger.mapper(lambda x: +.25*x**2 -.125*x**4,6,'legendre')
        bool_value = np.allclose(np.asarray(schrodinger.overall_hamiltonian(input_coefficients,0,1)),[0,0,1,0,0,0])
        inputs_2 = schrodinger.mapper(lambda x: .25*x**2 + 3j,5,'legendre') #checking to ensure complex #s don't throw an error
        self.assertTrue(bool_value)

    def test_mapper_fourier(self):
        input_coefficients_cos,input_coefficients_sin = schrodinger.mapper(lambda x: np.cos(x),5,'fourier')
        bool_value = np.allclose(input_coefficients_cos, [0,1,0])
        bool_value_2 = np.allclose(input_coefficients_sin, [0,0])
        self.assertTrue(bool_value)
        self.assertTrue(bool_value_2)

    def test_derivative_fourier(self):
        cos_in = [1,1,1,1]
        sin_in = [1,1,1]
        cos_out,sin_out = schrodinger.derivative_fourier(cos_in,sin_in)
        self.assertTrue(np.allclose(cos_out,[0,1,2,3]))
        self.assertTrue(np.allclose(sin_out,[-1,-2,-3]))

    def test_hamiltonian_fourier(self):
        cos_in = [1,1,1,1]
        sin_in = [1,1,1]
        cos_out, sin_out = schrodinger.hamiltonian_fourier(cos_in,sin_in,3,2)
        self.assertTrue(np.allclose(cos_out,[3,5,11,21]))
        self.assertTrue(np.allclose(sin_out,[5,11,21]))

    def test_overall_hamiltonian_fourier(self):
        cos_in,sin_in = schrodinger.mapper(lambda x: -np.cos(2*x)/4,5,'fourier')
        output_coefficients = schrodinger.overall_hamiltonian_fourier(cos_in,sin_in,0,-1)
        self.assertTrue(np.allclose(output_coefficients,[0,0,1,0,0]))
        cos_in,sin_in = schrodinger.mapper(lambda x: -np.sin(2*x) / 4, 11, 'fourier')
        output_coefficients = schrodinger.overall_hamiltonian_fourier(cos_in,sin_in,3,-1)
        self.assertTrue(np.allclose(output_coefficients,[0,0,0,0,0,0,0,.25,0,0,0]))

    def test_exception_handling(self):
        self.assertRaises(ValueError,schrodinger.mapper,lambda x: x**2, 6, 'fourier')
    




        


tests = unittest.TestLoader().loadTestsFromTestCase(TestSchrodinger)
unittest.TextTestRunner().run(tests)
        


if __name__ == '__main__':
    sys.exit(unittest.main())
