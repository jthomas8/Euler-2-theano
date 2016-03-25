#!/usr/local/bin/python3
import theano
from theano import shared
import math
import scipy.constants

# Constants
phi_cubed = shared(scipy.constants.golden ** 3)
last_even = theano.tensor.iscalar('last_even')
prev_summation = theano.tensor.iscalar('prev_summation')
new_summation = theano.tensor.iscalar('new_summation')

# Construct Theano Expression graph
new_summation = theano.tensor.floor((prev_summation + (last_even *  phi_cubed) + .5))

f = theano.function (
            inputs = [last_even, prev_summation],
            outputs = new_summation)
print(f(2, 2))
