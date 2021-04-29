# Here are many applications from numpy array


import numpy as np
import math
from numpy.linalg import matrix_power

def numpy_slicing(nplist, start, end):
    return nplist[start:end]


def numpy_update(nplist, ind, val):
    """ this will return updated nplist """
    if len(nplist) > ind:
        nplist[ind] = val
    return nplist

def numpyfied(numlist):
    """ turn python list into numpy list """
    return np.array(numlist)

def numpy_slice(numlist, start, end, skip) :
    new_list = []
    for i in range(start,end,skip):
        new_list.append(numlist[i])
    return np.array(new_list)
    
"""
Let's say you have this array, x:

>>> import numpy
>>> x = numpy.array([[ 1,  2,  3,  4,  5],
... [ 6,  7,  8,  9, 10],
... [11, 12, 13, 14, 15],
... [16, 17, 18, 19, 20]])
To get every other odd row, like you mentioned above:

>>> x[::2]
array([[ 1,  2,  3,  4,  5],
       [11, 12, 13, 14, 15]])
To get every other even column, like you mentioned above:

>>> x[:, 1::2]
array([[ 2,  4],
       [ 7,  9],
       [12, 14],
       [17, 19]])
Then, combining them together yields:

>>> x[::2, 1::2]
array([[ 2,  4],
       [12, 14]])
"""

def odd_arrays(numpy_array):
    """return odd number rows"""
    return numpy_array[1::2]

def reshaped_by_col(nums, col):
    ''' return rows with certain column '''
    if len(nums) % col != 0:
        return None
    return np.reshape(np.array(nums),(-1,col))

def squared(numlist):
    """ return squared numlist """
    return np.array(numlist)**2

def correct_rainfall(rainfall, day, correction):
    """ this modify nparray with day and correction """
    array = np.array(rainfall)
    array[:,day] += correction
    return array

def no_odds(nums):
    """ this uses where function
    to determine odds number and + 1 """
    array = np.array(nums)
    array[np.where((array % 2) != 0)] += 1
    return array

def abs_list(nums):
    """ this uses where function to
    find negative value and times by -1"""
    array = np.array(nums)
    array[np.where(array < 0)] *= -1
    return array

def filtered_average(scores, threshold):
    """
    function uses where function to find
    scores that greater than threshold and 
    get average. If there is no score greater 
    than threshold return 0.0
    """
    array = np.array(scores)
    if np.amax(array) < threshold:
        return 0.0
    return np.sum(array[np.where(array >= threshold)]) / len(array[np.where(array >= threshold)])

def npcrossing(value):
    '''
    this help cos_x_crossings function
    '''
    return np.cos(value)

def cos_x_crossings(begin, end, skip):
    """ it will generate range of values skipped by skip
    and calculate y-intercept value"""
    array = np.arange(begin, end, skip)
    values = npcrossing(array)
    times = values[:-1] * values[1:]
    times[np.where(times < 0)] -= skip
    return array[np.where(times < 0)]

def dotproduct(vector1, vector2):
    """
    this gives dotproduct of two vectors
    """
    if len(vector1) != len(vector2):
        return None
    return np.array(vector1).dot(np.array(vector2))

def euclidean_distance(vector1, vector2):
    """ this calculate euclidean distance
    by adding the sum difference and then taking a square root."""
    if len(vector1) != len(vector2):
        return None
    return math.sqrt(sum((np.array(vector1) - np.array(vector2))**2))

def e_approximation(x, n):
    """ at the end of function +1 is when n = 0
    this will calculate e value """
    values = np.arange(1,n+1)
    return sum(x ** values / np.cumprod(values)) + 1.0

def matrix_expo(matrix, limit):
    """ this will find exponent value which make max would not exceed limit"""
    npmat = np.array(matrix)
    power = 1
    check = np.amax(npmat)
    while check < limit:
        power += 1
        check = np.max(matrix_power(npmat,power))
    return power

def larger_multiplier(matrix, exp):
    """ this will find multiplier value m when sum of matrix * m
    exceeds power of matrix,exp"""
    npmat = np.array(matrix)
    if np.sum(matrix_power(npmat,exp)) % np.sum(npmat) != 0:
        return (np.sum(matrix_power(npmat,exp)) // np.sum(npmat)) + 1
    return np.sum(matrix_power(npmat,exp)) // np.sum(npmat)

def numpy_save(numlist, filetype):
    """ this will save numlist into npy or csv """
    if filetype == "csv":
        np.savetxt("newfile.csv", numlist, delimiter = ",")
    else:
        np.save("newfile", numlist)
        
    # np.load or np.loadtxt with .npy and .csv, delimiter = ","

def larger_sum(npfile1, npfile2):
    """ this will load nparray and find difference """
    nparr1 = np.load(npfile1)
    nparr2 = np.load(npfile2)
    if np.sum(nparr1) > np.sum(nparr2):
        print(f"{npfile1}: {(np.sum(nparr1) - np.sum(nparr2)):.4f}")
    else:
        print(f"{npfile2}: {(np.sum(nparr2) - np.sum(nparr1)):.4f}")
