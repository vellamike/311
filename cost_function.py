import numpy
import math

def error_function(predicted,actual):
    """
    Error function as described here:
    http://www.kaggle.com/c/see-click-predict-fix/details/evaluation

    g = sum((log(p_i + 1) - log(a_i + 1))^2)
    f = sqrt(g/n)
    
    """
    
    assert len(predicted) == len(actual)

    n = len(predicted)

    g = 0
    for p,a in zip(predicted,actual):
        g += (math.log(p + 1) - math.log(a+1))**2

    f = math.sqrt(g/n)

    return f
