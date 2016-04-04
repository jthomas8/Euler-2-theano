#!/usr/local/bin/python3
import theano
from theano import shared
import numpy as np
import scipy.constants

phi_cubed = shared(scipy.constants.golden ** 3)

def next_even_fib(previous_fib, max_value):
    return np.floor(previous_fib * phi_cubed + 0.5), theano.scan_module.until(np.floor(previous_fib * phi_cubed + 0.5) > max_value)

max_value = theano.tensor.scalar()

values, _ = theano.scan(next_even_fib,
                        outputs_info = theano.tensor.constant(2., dtype='float64'),
                        non_sequences = max_value,
                        n_steps = 1024)

summation = 2 + np.sum(values[0:-1])

f = theano.function([max_value], values)
print(f(4000000))

