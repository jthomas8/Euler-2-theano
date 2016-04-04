#!/usr/local/bin/python3
import theano
from theano import shared
import numpy as np
import scipy.constants

phi_cubed = shared(scipy.constants.golden ** 3)
summation = theano.tensor.iscalar('summation')
def next_even_fib(previous_fib, max_value):
    return np.floor(previous_fib * phi_cubed + 0.5), theano.scan_module.until(np.floor(previous_fib * phi_cubed + 0.5) > max_value)

max_value = theano.tensor.scalar()

values, _ = theano.scan(next_even_fib,
                        outputs_info = theano.tensor.constant(2., dtype='float64'),
                        non_sequences = max_value,
                        n_steps = 1024)
summation = theano.tensor.sum(values[0:-1]) + 2
f = theano.function([max_value], summation)


print(f(4000000))

