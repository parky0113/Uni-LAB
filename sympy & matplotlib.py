import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
from sympy.solvers import solve
import matplotlib.pyplot as plt
import math
# Never write file name as one of library names

def differentiate_equation(exp):
    """ 
    this will return differentiated function of exp 
    parse_expr converts the string s to a SymPy expression
    
    """
    return sp.diff(parse_expr(exp))

def n_differentiate_equation(exp, n):
    """differentiate n times"""
    deriv = parse_expr(exp)
    for _ in range(n):
        deriv = sp.diff(deriv)
    return deriv

def integrate_expression(exp):
    """ this function integrate exp """
    return sp.integrate(parse_expr(exp))

def definite_integral(expression, symbol, x_range):
    """
    this will find definite integrals
    which mean by getting area between two points
    according to function of exp
    to get numerical value try using sp.N
    e.g. sp.N(sp.integrate(parse_expr(expression), (symbol,x_range)))
    """
    return sp.integrate(parse_expr(expression), (symbol,x_range))

def check_intersection(exp1, exp2):
    """Solve this equation by doing exp1=exp2
    that will return intersections"""
    return len(sp.solve(parse_expr(exp1) - (parse_expr(exp2))))

def find_roots(exp):
    """this will find x which make exp 0 """
    return solve(parse_expr(exp))

def num_intersections(expressions):
    """ this will count intersection of each expression with others in list
    using nested loop"""
    count_list = []
    for i in range(len(expressions)):
        main = parse_expr(expressions[i])
        count = 0
        for j in range(len(expressions)):
            if i == j:
                continue
            count += len(sp.solve(main - (parse_expr(expressions[j]))))
        count_list.append(count)
    return count_list


def make_graph(y_vals):
    """ with given data, y_vals
    draw a line graph 
    returning plt obj is just return plt """
    plt.plot(y_vals)
    plt.show()
    return plt

def make_graph(x_vals, y_vals):
    """ this function will graph
    values given x_vals and y_vals """
    plt.plot(x_vals, y_vals)
    plt.show()
    return plt

def looping(my_func, interval, limit):
    """
    this function will generate graph
    with given functions and certain x_vals
    """
    x = 0.0
    while x < limit:
        plt.plot(x, my_func(x), 'r.')
        x += interval
    plt.show()
    return plt

def plot_legend(filename, x_axis, data):
    """
    this function will plot csv data
    with labels and certain data
    """
    df = pd.read_csv(filename)
    for label in data.keys():
        plt.plot(df[x_axis], df[label], label= data.get(label))
    plt.legend()
    plt.show()
    return plt

def gaussian(x, x0, k0, sigma):
    """ this will help function make_packet
    this is base function for gaussian """
    return (abs((1/((2*np.pi*(sigma**2))**(1/4))) \
    * (np.exp(1))**((-((x-(x0))**2)) / (4*(sigma**2))) \
    * (np.exp(1)**((1j)*(k0)*x))))**2

def make_packet(x0, k0, sigma):
    """ this function will draw graph
    gaussian packet wave idk how 
    but still it works"""
    xs = np.arange(0, 100.5, 0.5)
    plt.plot(xs, gaussian(xs, x0, k0, sigma))
    plt.show()
    return plt

"""
multiple conditions for np.where
arr = arr[np.where((condition1) & (condition2))]
"""

def sympy_grapher(expression, granularity=0.1, limits=[-10,10,-10,10]):
    """
    this function will graph function expression
    """
    x = sp.symbols('x')
    x_pts = np.arange(limits[0], limits[1]+granularity, granularity)
    expr = parse_expr(expression)
    f = sp.lambdify(x, expr, 'numpy') 
    y_arr = f(x_pts)
    plt.plot(x_pts, y_arr)
    plt.ylim([limits[2], limits[3]])
    plt.xlim([limits[0], limits[1]])
    plt.show()
    return plt

#limit때문에 시바 1시간을 죽쒓네...
