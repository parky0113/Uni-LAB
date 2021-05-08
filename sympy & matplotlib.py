import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
from sympy.solvers import solve
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
            else:
                count += len(sp.solve(main - (parse_expr(expressions[j]))))
        count_list.append(count)
    return count_list
