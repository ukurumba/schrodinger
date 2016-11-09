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

    def test_overall_hamiltonian_and_mapper(self):
    	input_coefficients = schrodinger.mapper(lambda x: +.25*x**2 -.125*x**4,6)
    	bool_value = np.allclose(np.asarray(schrodinger.overall_hamiltonian(input_coefficients,0,1)),[0,0,1,0,0,0])
    	inputs_2 = schrodinger.mapper(lambda x: .25*x**2 + 3j,5)
    	self.assertTrue(bool_value)



    	


tests = unittest.TestLoader().loadTestsFromTestCase(TestSchrodinger)
unittest.TextTestRunner().run(tests)
        


if __name__ == '__main__':
    sys.exit(unittest.main())
