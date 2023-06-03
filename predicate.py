import os
import sys
from numpy import number
import z3
import dis
import inspect
import uncompyle6
from io import StringIO
from lambda_symbolic_exec.lambda_expr_eval import *
from util_type import *
from util import *


def is_null_constant(c):
    return c is None or str(c) == 'nan'
def const_to_code(c):
    return "np.nan" if (c is None or str(c) == 'nan') else ("'{}'".format(c.replace("'","\\'")) if type(c) is str else c)

class Atomic(object):
    def __init__(self):
        pass

class Variable(Atomic):
    def __init__(self, v, values = [], typ='int'):
        self.v = v
        self.values = values
        self.typ = typ
    def eval(self, tup):
        return self.v
    def isnull(self, tup):
        return self.v is None
    def __str__(self):
        return 'v-' + str(self.v)

class Field(Atomic):
    def __init__(self, col):
        self.col = col
    def eval(self, tup):
        return tup[self.col]
    def isnull(self, tup):
        return tup[self.col].isnull
    def __str__(self):
        return '['+str(self.col)+']'
    
class Expr(Atomic):
    def __init__(self, expr, return_type=None):
        self.expr = expr
        self.return_type = return_type
    def eval(self, tup):
        return eval_lambda(self.expr, self.return_type, tup.values if type(tup) is Tuple else tup)
    def isnull(self, tup):
        fields = get_column_used_from_lambda(self.expr)
        if len(fields) == 0:
            return False
        else:
            return z3.Or(*[False if isinstance(tup[f], Constant) else tup[f].isnull for f in fields])
    def __str__(self):
        x = get_lambda_code(self.expr)
        return x[x.index(":")+1:]

from datetime import datetime
from dateutil import parser
class Constant(Atomic):
    def __init__(self, v, typ=None):
        self.v = v
        self.typ = typ
        if self.typ is None:
            self.typ = 'str' if type(self.v) is str else 'int'
    def eval(self, tup):
        # need update
        if self.typ in ['date','datetime', 'datetime64[ns]']:
            dt_time = parser.parse(str(self.v))
            return 10000*dt_time.year + 100*dt_time.month + dt_time.day
        if type(self.v) == str or self.typ == 'object':
            return str(self.v)#'"{}"'.format(self.v)
        elif type(self.v) is list:
            return self.v
        else:
            return int(self.v)
    def isnull(self, tup=None):
        return False
    def __str__(self):
        return 'c-'+str(self.v)


class String(Atomic):
    def __init__(self, v):
        self.v = v
    def eval(self, tup):
        return self.v
    def isnull(self, tup):
        return False
    def __str__(self):
        return 's-'+str(self.v)

operator_z3_map = {'>':lambda x,y: x>y, '<':lambda x,y: x<y, '>=':lambda x,y: x>=y, '<=':lambda x,y: x<=y, '==':lambda x,y: x==y, '!=':lambda x,y: x!=y,\
                '+':lambda x,y: x+y, '-':lambda x,y: x-y, '*':lambda x,y:x*y, '/':lambda x,y:x/y,\
                'in':lambda x,y: any([x==y1 for y1 in y]), 'min':lambda x,y: min(x,y), 'max':lambda x,y: max(x, y), \
                'subset': lambda x,y: z3.Contains(y, x)}

class UnaryOp(object):
    def __init__(self, v):
        self.v = v
    def __str__(self):
        return 'Not({})'.format(self.v)
class Not(UnaryOp):
    def eval(self, tup):
        return z3.Not(self.v.eval(tup))
    def isnull(self, tup):
        return False
    def __str__(self):
        return 'not({})'.format(self.v)
class IsNULL(UnaryOp):
    def eval(self, tup):
        return self.v.isnull(tup)
    def __str__(self):
        return 'isnull({})'.format(self.v)
class IsNotNULL(UnaryOp):
    def eval(self, tup):
        return self.v.isnull(tup)==False
    def __str__(self):
        return 'isnotnull({})'.format(self.v)        

class BinOp(object):
    def __init__(self, lh, op, rh):
        self.lh = lh
        self.op = op
        self.rh = rh
    def eval(self, tup):
        lh = zeval(self.lh, tup)
        rh = zeval(self.rh, tup)
        if isinstance(lh, Value):
            lh_null = lh.isnull
            lh_v = lh.v
        else:
            lh_null = False
            lh_v = lh
        if isinstance(rh, Value):
            rh_null = rh.isnull 
            rh_v = rh.v
        else:
            rh_null = False
            rh_v = rh
        if self.op == '==':
            return z3.Or(z3.And(lh_null, rh_null), z3.And(lh_null==False, rh_null==False, operator_z3_map[self.op](lh_v, rh_v)))
        else: 
            return z3.If(z3.Or(lh_null, rh_null), False, operator_z3_map[self.op](lh_v, rh_v))
    def isnull(self, tup):
        return z3.Or(self.lh.isnull(tup), self.rh.isnull(tup))
    def __str__(self):
        return '({} {} {})'.format(self.lh, self.op, self.rh)

class BoolOp(object):
    def __init__(self, lh, rh):
        self.lh = lh
        self.rh = rh
    def isnull(self, tup):
        return False

class And(BoolOp):
    def eval(self, tup):
        lh = zeval(self.lh, tup)
        rh = zeval(self.rh, tup)
        return z3.And(lh, rh)
    def __str__(self):
        return '({} && {})'.format(self.lh, self.rh)
    def split(self):
        ret = []
        if isinstance(self.lh, And):
            ret.extend(self.lh.split())
        else:
            ret.append(self.lh)
        if isinstance(self.rh, And):
            ret.extend(self.rh.split())
        else:
            ret.append(self.rh)
        return ret

class Or(BoolOp):
    def eval(self, tup):
        lh = zeval(self.lh, tup)
        rh = zeval(self.rh, tup)
        return z3.Or(lh, rh)
    def __str__(self):
        return '({} || {})'.format(self.lh, self.rh)

def AllAnd(*args):
    if len(args)==0:
        return True
    ret = args[0]
    for x in args[1:]:
        ret = And(ret, x)
    return ret

def AllOr(*args):
    if len(args)==0:
        return True
    ret = args[0]
    for x in args[1:]:
        ret = Or(ret, x)
    return ret

def get_lambda_code(expr):
    expr_str = expr
    lambda_code = expr_str[expr_str.find('lambda '):]
    return lambda_code


def get_columns_used(expr):
    if isinstance(expr, BoolOp):
        return get_columns_used(expr.lh) + get_columns_used(expr.rh)
    elif isinstance(expr, BinOp):
        return get_columns_used(expr.lh) + get_columns_used(expr.rh)
    elif isinstance(expr, UnaryOp):
        return get_columns_used(expr.v)
    elif isinstance(expr, Field):
        return [expr.col]
    elif isinstance(expr, Expr):
        var = get_lambda_varibale_name(expr.expr) 
        expr_func = eval(expr.expr)
        instrs = [instr for instr in dis.Bytecode(expr_func.__code__)]
        cols = []
        for i,instr in enumerate(instrs):
            if instr.opname in ["LOAD_FAST",'LOAD_GLOBAL'] and instr.argval == var and i < len(instrs)-1 and instrs[i+1].opname == "LOAD_CONST":
                cols.append(instrs[i+1].argval)
        return cols
    else:
        return []



def expr_contain_column(expr, cols):
    if isinstance(expr, BoolOp) or isinstance(expr, BinOp):
        return expr_contain_column(expr.lh, cols) or expr_contain_column(expr.rh, cols)
    elif isinstance(expr, UnaryOp):
        return expr_contain_column(expr.v, cols)
    elif isinstance(expr, Expr):
        cols_used = get_columns_used(expr)
        if set(cols_used)& set(cols):
            return True
        else:
            return False
    elif isinstance(expr, Field):
        return expr.col in cols
    else:
        return False

def get_filter_removing_unused(expr, columns_not_used): # TODO
    if isinstance(expr, BoolOp) or (isinstance(expr, Not) and isinstance(expr.v, BoolOp)):
        if isinstance(expr, Not):
            backup_expr  = expr
            expr = expr.v
        else:
            backup_expr = None
        if isinstance(expr.lh, BoolOp):
            lh = get_filter_removing_unused(expr.lh, columns_not_used)
        elif expr_contain_column(expr.lh, columns_not_used):
            lh = True
        else:
            lh = expr.lh
        if isinstance(expr.rh, BoolOp):
            rh = get_filter_removing_unused(expr.rh, columns_not_used)
        elif expr_contain_column(expr.rh, columns_not_used):
            rh = True
        else:
            rh = expr.rh
        
        if backup_expr is not None:
            if isinstance(expr, And):
                return Not(And(lh, rh))
            elif isinstance(expr, Or):
                return Not(Or(lh, rh))
        else:
            if isinstance(expr, And):
                return And(lh, rh)
            elif isinstance(expr, Or):
                return Or(lh, rh)
    elif expr_contain_column(expr, columns_not_used):
        return True
    else:
        return expr


def replace_lambda_in_lambda(var, col, f, col_lambda):
    ret = f
    col_lambda_body = col_lambda[col_lambda.find(':')+1:].lstrip()
    var_name = get_lambda_variable_names(col_lambda)[0]
    col_lambda_body = col_lambda_body.replace('{}['.format(var_name), '{}['.format(var))
    if '{}["{}"]'.format(var, col) in f:
        ret = ret.replace('{}["{}"]'.format(var, col), '({})'.format(col_lambda_body))
    if "{}['{}']".format(var, col) in f:
        ret = ret.replace("{}['{}']".format(var, col), '({})'.format(col_lambda_body))
    return ret

def get_filter_replacing_field(expr, column_replace, para=[]):
    if isinstance(expr, BoolOp):
        if isinstance(expr, And):
            return And(get_filter_replacing_field(expr.lh, column_replace, para), get_filter_replacing_field(expr.rh, column_replace, para))
        else : 
            return Or(get_filter_replacing_field(expr.lh, column_replace, para), get_filter_replacing_field(expr.rh, column_replace, para))
    elif isinstance(expr, BinOp):
        if isinstance(expr.lh, Field) or isinstance(expr.lh, Expr):
            lh = get_filter_replacing_field(expr.lh, column_replace, para)
        else:
            lh = expr.lh
        if isinstance(expr.rh, Field) or isinstance(expr.rh, Expr):
            rh = get_filter_replacing_field(expr.rh, column_replace, para)
        else:
            rh = expr.rh
        return BinOp(lh, expr.op, rh)
    elif isinstance(expr, Field):
        if expr.col in column_replace:
            # case 1: field + map
            if type(column_replace[expr.col]) in [str, int]:
                return Field(column_replace[expr.col])
            elif isinstance(column_replace[expr.col], Variable):
                return column_replace[expr.col]
            # case 2: field +lambda
            else:
                new_col_lambda = column_replace[expr.col]
                #print(new_col_lambda)
                return_type = new_col_lambda.return_type
                new_col_lambda = get_lambda_code(new_col_lambda.expr)
                
                if "fill_value" in new_col_lambda:
                    number_str = str(para[1])
                    col_str = '"' + str(para[0]) + '"'
                    new_col_lambda = new_col_lambda.replace("fill_value", number_str)
                    new_col_lambda = new_col_lambda.replace("col", col_str)
                    return_type = get_variable_type(param[1])
                    print(new_col_lambda)
                return Expr(new_col_lambda, return_type)
        else:
            return expr
    elif isinstance(expr, Expr):
        col_used = get_column_used_from_lambda(expr.expr)
        if set(col_used) & set(column_replace):
            # case 3: lambda + map
            if all([col_ not in column_replace or not isinstance(column_replace[col_], Expr) for col_ in col_used]):
                lambda_str = expr.expr
                lambda_var = get_lambda_varibale_name(expr.expr)
                modified = False
                for old_col,new_col in column_replace.items():
                    if '{}["{}"]'.format(lambda_var, old_col) in lambda_str:
                        lambda_str = lambda_str.replace('{}["{}"]'.format(lambda_var, old_col), '{}["{}"]'.format(lambda_var, new_col))
                        modified = True
                    elif "{}['{}']".format(lambda_var, old_col) in lambda_str:
                        lambda_str = lambda_str.replace("{}['{}']".format(lambda_var, old_col), "{}['{}']".format(lambda_var, new_col))
                        modified = True
                if modified == True:
                    return Expr(lambda_str, expr.return_type)
                else:
                    return expr
            else:
                var = get_lambda_varibale_name(expr.expr)
                col_lambda = ''
                col_used_ = ''
                for col_ in col_used:
                    if col_ in column_replace:
                        col_lambda = column_replace[col_].expr
                        col_used_ = col_
                new_col_lambda = replace_lambda_in_lambda(var, col_used_, expr.expr, col_lambda)
                return Expr(new_col_lambda, expr.return_type)
        else:
            return expr
    elif isinstance(expr, UnaryOp):
        newv = get_filter_replacing_field(expr.v, column_replace, para)
        if isinstance(expr, Not):
            return Not(newv)
        elif isinstance(expr, IsNULL):
            return IsNULL(newv)
        elif isinstance(expr, IsNotNULL):
            return IsNotNULL(newv)
    else:
        return expr



def get_filter_replacing_for_unpivot(expr, columns_in_idvar, value_name, var_name, value_vars):
    if isinstance(expr, BoolOp):
        if isinstance(expr, And):
            return And(get_filter_replacing_for_unpivot(expr.lh,  columns_in_idvar, value_name, var_name, value_vars),get_filter_replacing_for_unpivot(expr.rh,  columns_in_idvar, value_name, var_name, value_vars))
        else : 
            return Or(get_filter_replacing_for_unpivot(expr.lh, columns_in_idvar, value_name, var_name, value_vars),get_filter_replacing_for_unpivot(expr.rh, columns_in_idvar, value_name, var_name, value_vars))
    elif isinstance(expr, BinOp):
        if all([col in columns_in_idvar for col in get_columns_used(expr)]): #expr.lh.col in columns_in_idvar:
            return expr
        elif any([col==value_name for col in get_columns_used(expr)]): #expr.lh.col == value_name:
            pred = []
            col_to_replace = list(filter(lambda col:col==value_name, get_columns_used(expr)))[0]
            for v in value_vars:
                #pred.append(BinOp(Field(v), expr.op, expr.rh))
                pred.append(get_filter_replacing_field(expr, {col_to_replace:v}))
            return AllOr(*pred)
        elif any([col==var_name for col in get_columns_used(expr)]): # expr.lh.col == var_name
            # we ignore the projection
            col_to_replace = list(filter(lambda col:col==var_name, get_columns_used(expr)))
            return get_filter_removing_unused(expr, col_to_replace)
            #return True
        else:
            return expr
    elif isinstance(expr, UnaryOp):
        expr.v = get_filter_replacing_for_unpivot(expr.v, columns_in_idvar, value_name, var_name, value_vars)
        return expr
    else:
        return expr





def get_filter_replacing_unused_for_pivot(expr, unused_column, header_col, value_col):  
    # split into conjunctions where each contain one unused column
    if isinstance(expr, And):
        split_pred = expr.split()
    else:
        split_pred = [expr]
    print("split pred = {}".format(",".join([str(xx) for xx in split_pred])))
    print("unused_cols = {}".format(unused_column))
    unused_column = set(unused_column)
    split_column_used = [set(get_columns_used(e)) for e in split_pred]
    good_split = all([len(cols.intersection(unused_column)) <= 1 for cols in split_column_used])
    assert(good_split)
    ret = []
    other = []
    for i in range(len(split_pred)):
        if len(split_column_used[i].intersection(unused_column)) == 0:
            other.append(split_pred[i])
            continue
        header_value = list(split_column_used[i])[0]
        new_pred = get_filter_replacing_field(split_pred[i], {header_value:value_col})
        ret.append(And(new_pred, BinOp(Field(header_col), '==', Constant(header_value, 'str'))))
    other = AllAnd(*other)
    if len(ret) == 0:
        return other
    else:
        return AllOr(*[And(other, r) for r in ret])




def get_filter_replacing_with_schema(expr, orig_col, new_cols):
    if isinstance(expr, BoolOp):
        if isinstance(expr, And):
            return And(get_filter_replacing_with_schema(expr.lh,  orig_col, new_cols),get_filter_replacing_with_schema(expr.rh,  orig_col, new_cols))
        else : 
            return Or(get_filter_replacing_with_schema(expr.lh, orig_col, new_cols),get_filter_replacing_with_schema(expr.rh,  orig_col, new_cols))
    elif isinstance(expr, BinOp):
        if isinstance(expr.rh, Constant) and expr.rh.v == 1:
            if expr.lh.col in list(new_cols.keys()):
                rh_val = expr.lh.col
                return BinOp(Field(orig_col), expr.op, Constant(rh_val, 'str'))
            else:
                return expr
        elif isinstance(expr.rh, Constant) and expr.rh.v == 0:
            return True
        else:
            return expr
    elif isinstance(expr, UnaryOp):
        newv = get_filter_replacing_with_schema(expr.v, orig_col, new_cols)
        if isinstance(expr, Not):
            return Not(newv)
        elif isinstance(expr, IsNULL):
            return IsNULL(newv)
        elif isinstance(expr, IsNotNULL):
            return IsNotNULL(newv)
    else:
        return expr



def detect_binop_value_in_lambda(instrs, pos):
    for i,instr in reversed(list(enumerate(instrs))):
        if i >= pos:
            continue
        if i >= 2 and instrs[i-2].opcode == 124 and instrs[i-1].opcode == 100 and instr.opcode == 25: 
            return Field(instrs[i-1].argval), i-2
        elif instr.opcode == 100:
            return Constant(instr.argval), i
    return None, -1

def get_predicate_from_lambda(f, var_field_map={}): 
    f_func = eval(f)
    ret = []
    instrs = [instr for instr in dis.Bytecode(f_func)]
    for i,instr in enumerate(instrs):
        print(instr)
        if instr.opcode==107:
            op1,pos = detect_binop_value_in_lambda(instrs, i)
            if pos == -1:
                return []
            op2,pos2 = detect_binop_value_in_lambda(instrs, pos)
            if pos2 == -1:
                return []
            if instr.argval == 'in' and any([isinstance(op, Constant) and type(op.v) is str for op in [op1,op2]]):
                ret.append(BinOp(op2, 'subset', op1))
            else:
                ret.append(BinOp(op2, instr.argval, op1))
    return ret


def get_pandas_filter_lambda_str(expr, column_var_map):
    if isinstance(expr, And):
        return get_pandas_filter_lambda_str(expr.lh, column_var_map) + " && " + get_pandas_filter_lambda_str(expr.rh, column_var_map)
    elif isinstance(expr, Or):
        return get_pandas_filter_lambda_str(expr.lh, column_var_map) + " || " + get_pandas_filter_lambda_str(expr.rh, column_var_map)
    elif isinstance(expr, BinOp):
        return "(" + get_pandas_filter_lambda_str(expr.lh, column_var_map) + " {} ".format(expr.op) + get_pandas_filter_lambda_str(expr.rh, column_var_map) + ")"
    elif isinstance(expr, UnaryOp):
        if isinstance(expr, Not):
            return "(!{})".format(get_pandas_filter_lambda_str(expr.v, column_var_map))
        elif isinstance(expr, IsNULL):
            return "pd.isnull({})".format(get_pandas_filter_lambda_str(expr.v, column_var_map))
        elif isinstance(expr, IsNotNULL):
            return "!pd.isnull({})".format(get_pandas_filter_lambda_str(expr.v, column_var_map))
    elif isinstance(expr, Variable):
        return str(column_var_map[expr.v])
    elif isinstance(expr, Constant):
        return str(expr.v)
    elif isinstance(expr, String):
        return str(expr.v)
    elif isinstance(expr, Field):
        return "x[{}]".format(expr.col)
    elif type(expr) is bool:
        return str(expr)
    else:
        print("CANNOT HANDLE {} {}".format(expr, type(expr)))


def determine_join_pushdown(expr, columns_used_in_left, columns_used_in_right):
    if (len(columns_used_in_left)> 0 and len(columns_used_in_right) > 0):
        if isinstance(expr, BinOp):
            if(isinstance(expr.lh, Field) and isinstance(expr.lh, Field)):
                if expr.lh.col in columns_used_in_left and expr.rh.col in columns_used_in_right or expr.lh.col in columns_used_in_right and expr.rh.col in columns_used_in_left:
                    print("Cannot push down ")
                    exit(0)
        elif isinstance(expr, Or):
            if(len(set(get_columns_used(expr.lh)).intersection(columns_used_in_left))>0 and len(set(get_columns_used(expr.rh)).intersection(columns_used_in_right))>0):
                print("Cannot push down")
                exit(0)

def check_predicate_equal (pred1, pred2, tup):
    return check_always_hold(zeval(pred1, tup)==zeval(pred2, tup))



