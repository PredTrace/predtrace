from re import A
import z3
import dis
import pandas as pd
from predicate import *
from util_type import *
from lambda_symbolic_exec.lambda_expr_eval import *

class AstRefKey:
    def __init__(self, n):
        self.n = n
    def __hash__(self):
        return self.n.hash()
    def __eq__(self, other):
        return self.n.eq(other.n)
    def __repr__(self):
        return str(self.n)

def askey(n):
    assert isinstance(n, z3.AstRef)
    return AstRefKey(n)

def z3_simplify(expr):
    if type(expr) in [bool, int, str, float]:
        return expr
    else:
        return z3.simplify(expr)

def get_vars_from_formula(f):
    r = set()
    def collect(f):
      if z3.is_const(f): 
          if f.decl().kind() == z3.Z3_OP_UNINTERPRETED and not askey(f) in r:
              r.add(askey(f))
      else:
          for c in f.children():
              collect(c)
    collect(f)
    return r

def to_pandas_table(table):
    values = {k:[] for k,v in table[0].values.items()}
    for tup in table:
        for k,v in tup.values.items():
            values[k].append(v)
    return pd.DataFrame(values)

lambda_type = type(lambda x: x)

def generate_symbolic_table(table_name, schema, Ntuples):
    ret = []
    for i in range(Ntuples):
        t = create_tuple({k:get_symbolic_value_by_type(v, '{}-tup-{}-{}'.format(table_name, i+1, k)) for k,v in schema.items()})
        ret.append(t)
    return ret

# aggr_func: lambda
def compute_aggregate(aggr_func, a_v, cond, b_v): # a is previous value; b is new element
    assert(isinstance(a_v, Value))
    assert(isinstance(b_v, Value))
    
    a = a_v.v
    b = b_v.v
    if aggr_func == 'sum' or aggr_func == 'mean' or aggr_func == 'avg':
        ret = a+b
    elif aggr_func == 'max':
        print("a = {} / {}, b = {} / {}".format(a,b, type(a), type(b)))
        ret = z3.If(a>b, a, b)
    elif aggr_func == 'min':
        ret = z3.If(a<b, a, b)
    elif aggr_func == 'count':
        ret = a+1
        b = 1
    elif aggr_func == 'unique':
        ret = a+b
    else: #if type(aggr_func) == lambda_type:
        #ret = aggr_func(a, b)
        ret = eval_lambda_other(aggr_func, None, a, b).v
    #else:
    #    assert(False)
    # a = null, b = null --> ?, True
    # a = null, b != null --> b.v, False
    # a != null, b = null --> a.v, False
    # a != null, b != null --> aggr(a,b), False 
    # print("a = {}, a.isnull = {}, b = {}, b.isnull = {}, cond = {}".format(a if type(a) is int else z3.simplify(a), \
    #                 a_v.isnull if type(a_v.isnull) is bool else z3.simplify(a_v.isnull), \
    #                     b, b_v.isnull, z3.simplify(cond)))
    return Value(z3.If(a_v.isnull, b, z3.If(cond, ret, a)), z3.And(a_v.isnull, cond==False))

def compute_aggregate_row(aggr_func, a_v, cond, b_v):
    a = {k:ax.v for k,ax in a_v.items()} if type(a_v) is dict else a_v.v
    b = {k:bx.v for k,bx in b_v.items()}
    ret = eval_lambda_other(aggr_func, None, a, b).v
    if type(a_v) is dict:
        return {k:Value(z3.If(cond, retv_, a[k].v)) for k,retv_ in ret.items()}
    else:
        return Value(z3.If(cond, ret, a))

def compute_aggregate_checking_null(aggr_func, a_v, a_v_null, cond, b_v, b_v_null):
    ret = aggr_func(a_v, b_v)
    is_null = z3.And(a_v_null, z3.Or(cond==False, b_v_null))
    if type(a_v) is tuple or type(a_v) is list:
        return [z3.If(cond, ret[i], a_v[i]) for i in range(len(a_v))], is_null
    else:
        return z3.If(cond, ret, a_v), is_null

# b_v is an row
def compute_aggregate_udf1(aggr_func, left_row, a_v, cond, b_v): # a is previous value; b is new row
    a = a_v
    b = b_v
    ret = aggr_func(left_row, a, b)
    if type(a) is tuple or type(a) is list:
        return [z3.If(cond, ret[i], a[i]) for i in range(len(a))]
    else:
        return z3.If(cond, ret, a) 
         

def check_always_hold(expr,debug_vars=[],eval_exprs={}):
    #print("CHECKING: {}".format(z3.simplify(expr)))
    variables = get_vars_from_formula(expr)
    additional_vars = []
    table_vars = []
    for v in variables:
        if str(v).startswith('additional_'):
            additional_vars.append(v.n)
        else:
            table_vars.append(v.n)

    solver = z3.Solver()
    solver.push()
    if len(additional_vars) == 0:
        solver.add(z3.Not(expr))
    else:
        solver.add(z3.Not(z3.Exists(additional_vars, z3.ForAll(table_vars, expr))))

    if solver.check() != z3.unsat:
        print("Solver failed!")
        return False
    else:
        solver.pop()
        return True

