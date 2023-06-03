import z3
import dis
import pandas as pd
#from predicate import *
import random

additional_var_count = 0

valid_data_types = ['int', 'string', 'str']

# schema of t is given
# verifier: output_filter1(t) == (under constraint(t)) output_filter2(t) 
# given output_filter1, constraint, output_filter2, exam the above formula is correct or not
# unique constraint, we already proved small model property (|t| = 2)
# t': symbolic table (row1: {col1: v-1, col2: v-2}, row2: {col1: v-3, col2: v-4})
# t': col1    col2
#     v-1     v-2
#     v-3     v-4
# output_filter1(t') == (under constraint(t')) output_filter2(t')
# output_filter1(t') (row1: expr1(v-1==1), row2: expr2(v-3==1)), assume output_filter1=(col1==1)
# output_filter2(t') (row1: expr1'(v-2==2), row2: expr2'(v-4==2)),assume output_filter2=(col2==2)
# expr1==expr1', expr2==expr2'
# (v-1==1)==(v-2==2), ..
# constraint(t'), FD: col2=col1+1 (col1->col2)
# v-2==v-1+1, v-4==v-3+1 
##  constraint'(t'): unique(col1), --> v-1!=v-3
# output_filter1(t') == (under constraint(t')) output_filter2(t')
# (v-2==v-1+1, v-4==v-3+1) -> (v-1==1)==(v-2==2) && (v-3==1)==(v-4==2)

# df1 (input_filter) -> op -> df2 (output_filter) -> ....
# take snapshot of df2
# df2' = df2[output_filter_old]
# df2'_row1 : col1==1 & col2==1
# df2'_row2 : col1==2 & col2==2
# output_filter = (col1==1 & col2==1) | (col1==2 & col2==2) | .... # NO NEED!
# df2'_rowx : col1==v1 & col2==v2
# input_filter = cola==f(v1) & colb==f(v2)
# input_filter: (cola==f(1) & colb==f(1)) | (cola==f(2) & colb==f(2))

class Value(object): # v-1, v-2, v-3, v-4
    def __init__(self, value, isnull=False):
        self.v = value
        self.isnull = isnull
    def equals(self, v_):
        if isinstance(v_, Value):
            return z3.Or(z3.And(self.isnull==True, v_.isnull==True), z3.And(self.isnull==False, v_.isnull==False, self.v==v_.v))
        else:
            return self.v == v_
    def isnull(self):
        return self.isnull
    def __str__(self):
        if self.isnull==True:
            return 'null'
        elif self.isnull==False:
            return str(self.v)
        else:
            return "{}({})".format(self.v, self.isnull)

# t' row1 {col1: v-1, col2:v-2}
# Tuple (values={'col1': Value(v-1), 'col2': Value(v-2)})
# after running on output_filter1
# exist_cond, after filter (output_filter1: col1==1)
# Tuple.exist_cond = v-1==1
class Tuple(object):
    def __init__(self, values, exist_cond=True, count=1):
        self.values = values # {col_name: Value}
        self.exist_cond = exist_cond
        self.count = count
    def eval_exist_cond(self):
        return zeval(self.exist_cond, self.values)
    def __getitem__(self, col):
        return self.values[col]
    def __str__(self):
        c = self.eval_exist_cond()
        return 'values = [{}], exist_cond = {}'.format(','.join(['{}:{}'.format(k,v) for k,v in self.values.items()]), c if type(c) is bool else z3.simplify(c))

def zeval(expr, tup):
    if hasattr(expr, 'eval'):
        return expr.eval(tup)
    else:
        return expr

def get_init_value_by_type(typ):
    if typ == 'int':
        return 0
    elif typ in ['str','object','string']:
        return ''
    else:
        return 0


def create_value(v, typ):
    return Value(v) if v is not None else Value(get_init_value_by_type(k), True)
    
def create_tuple(values, exist_cond=True, count=1):
    if all([isinstance(v, Value) for k,v in values.items()]):
        return Tuple(values, exist_cond, count)
    else:
        return Tuple({k:Value(v) if v is not None else Value(get_init_value_by_type(k), True) for k,v in values.items()}, exist_cond, count)

def getv(v):
    if isinstance(v, Value):
        return v.v
    elif hasattr(v, 'v'):
        return v.v
    else:
        return v
def get_isnull(v):
    if isinstance(v, Value):
        return v.isnull
    else:
        return False

def get_init_value_by_type(typ):
    if typ in ['float','int', 'datetime','date']:
        return 0
    elif typ in ['str','string']:
        return ''
    else:
        print("Type {} cannot be handled".format(typ))
        assert(False)
        
def get_variable_type(var):
    if type(var) in [int, float]:
        return 'int'
    elif type(var) in [str]:
        return 'str'
    elif isinstance(var, z3.FuncDeclRef):
        if str(f.range()) == 'Int':
            return 'int'
        else:
            return 'str'
    elif isinstance(var, z3.ArithRef) and var.is_int(): # int type
        return 'int'
    elif isinstance(var, z3.SeqRef): # string type
        return 'string'
    elif hasattr(var, 'typ'):
        return var.typ
    else:
        print(var)
        print(type(var))
        assert False, 'type not supported'
    return var

def get_new_variable_by_type(typ, vname='some-value'):
    if typ == 'int':
        return z3.Int(vname)
    elif typ in ['str','string']:
        return z3.String(vname)
    else:
        assert(False)

def get_random_value_by_type(typ):
    if typ == 'int':
        return random.randint(1, 100)

def get_symbolic_value_by_type(typ, vname):
    if typ in ['int','float','datetime']:
        return z3.Int(vname)
    elif typ in ['str','string']:
        return z3.String(vname)
    else:
        assert(False, 'type not supported')


def get_lambda_varibale_name(expr):
    expr_str = expr
    var = expr_str[expr_str.find('lambda ')+len('lambda '):expr_str.find(':')]
    return var

def get_lambda_variable_names(expr):
    expr_str = expr
    var = expr_str[expr_str.find('lambda ')+len('lambda '):expr_str.find(':')].split(',')
    return [x.replace(' ','') for x in var]

def get_column_used_from_lambda(f):
    f_func = eval(f)
    ret = []
    instrs = [instr for instr in dis.Bytecode(f_func)]
    for i,instr in enumerate(instrs):
        if instr.opcode==100 and i < len(instrs)-1 and instrs[i+1].opcode == 25 and i > 0 and instrs[i-1].opcode == 124: # LOAD_CONST followed by BINARY_SUBSTR
            ret.append(instr.argval)
    return ret

def convert_type_to_z3_types(typ):
    if typ in ['date','datetime']:
        return 'int'
    elif typ in ['float']:
        return 'int'
    elif typ in valid_data_types:
        return typ
    else:
        print("type {} not supported".format(typ))
        assert(False)
        
