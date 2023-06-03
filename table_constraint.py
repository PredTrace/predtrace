# Uniqueness constraint
# exist constraint ?
# Functional dependency 
#from msilib import Table
from numpy import column_stack
from parted import Constraint
from interface import *
from predicate import *
from util import *
import sys
from lambda_symbolic_exec.lambda_expr_eval import *

class TableConstraint(object):
    pass

class UniqueConstraint(TableConstraint):
    def __init__(self, columns):
        self.columns = columns
    def eval(self, tuples):
        ret = []
        for i in range(0, len(tuples)):
            for j in range(i+1, len(tuples)):
                # TODO: encode isnull
                ret.append(z3.Or([tuples[i][col].v!=tuples[j][col].v for col in self.columns]))
        return z3.And(*ret) 
    def __str__(self):
        return "UniqueConstraint({})".format(self.columns)

class DomainConstraint(TableConstraint):
    def __init__(self, constraints):
        self.constraints = constraints
    def eval(self, tuples):
        ret = []
        for tup in tuples:
            for c, v in self.constraints.items():
                if type(v) is list:
                    ret.append(z3.And(tup[c].v>=v[0], tup[c].v<=v[1])) # close for now
                elif type(v) is set:
                    ret.append(z3.Or(*[tup[c].v==v1 for v1 in v]))
        return z3.And(*ret)
    def __str__(self):
        return "DomainConstraint({})".format(self.constraints)

class FunctionalDependency(TableConstraint):
    def __init__(self, dependent_column, other_columns, expr):
        self.dependent_column = dependent_column
        self.other_columns = other_columns
        self.expr = expr
    def eval(self, tuples):
        ret = []
        for tup in tuples:
            lambda_ret = eval_lambda(self.expr, None, tup.values)
            ret.append(tup.values[self.dependent_column].v == getv(lambda_ret))
        return z3.And(*ret)
    def __str__(self):
        return "FunctionalDependency: {} = {}".format(self.other_columns, self.expr)

class FilterNonEmpty(TableConstraint): # table after self.pred is not empty
    def __init__(self, pred):
        self.pred = pred
    def eval(self, tuples):
        ret = []
        for tup in tuples:
            ret.append(z3.And(tup.eval_exist_cond(), self.pred.eval(tup)))
        return z3.Or(*ret)
    
class ColumnSorted(TableConstraint):
    def __init__(self, columns, order='desc'):
        self.columns = columns
        self.ordre = order
    def eval(self, tuples):
        ret = []
        for i in range(1, len(tuples)):
            if self.ordr == 'desc':
                if len(self.columns) == 1:
                    cond = tuples[i-1].values[self.columns[0]].v>=tuples[i].values[self.columns[0]].v
                else:
                    cond = tuples[i-1].values[self.columns[0]].v>tuples[i].values[self.columns[0]].v
                    acc = tuples[i-1].values[self.columns[0]].v==tuples[i].values[self.columns[0]].v
                    for j in range(1, len(self.columns)):
                        cond = z3.Or(cond, z3.And(acc, tuples[i-1].values[self.columns[j]].v>tuples[i].values[self.columns[j]].v))
                        acc = z3.And(acc, tuples[i-1].values[self.columns[j]].v==tuples[i].values[self.columns[j]].v)
                    cond = z3.Or(cond, acc)
                ret.append(cond)
            else:
                if len(self.columns) == 1:
                    cond = tuples[i-1].values[self.columns[0]].v<=tuples[i].values[self.columns[0]].v
                else:
                    cond = tuples[i-1].values[self.columns[0]].v<tuples[i].values[self.columns[0]].v
                    acc = tuples[i-1].values[self.columns[0]].v==tuples[i].values[self.columns[0]].v
                    for j in range(1, len(self.columns)):
                        cond = z3.Or(cond, z3.And(acc, tuples[i-1].values[self.columns[j]].v<tuples[i].values[self.columns[j]].v))
                        acc = z3.And(acc, tuples[i-1].values[self.columns[j]].v==tuples[i].values[self.columns[j]].v)
                    cond = z3.Or(cond, acc)
                ret.append(cond)
        return z3.And(*ret)

def get_constraints_from_init_table(df):
    constraints = []
    domain_dict = {}
    for col in df.columns:
        # get unique constriant
        if df[col].nunique() == df.shape[0]:
            constraints.append(UniqueConstraint([col]))
        # get domain constraint
        unique = df[col].unique()
        if(len(unique) < 10) or str(df[col].dtype) == 'object':
            domain_dict[col] = set(unique)
        else:
            min_v = df[col].min()
            max_v = df[col].max()
            domain_dict[col] = [min_v, max_v]
    constraints.append(DomainConstraint(domain_dict))
    return constraints

def get_unique_constraints(lst):
    return list(filter(lambda x: isinstance(x, UniqueConstraint), lst))

def get_fd_constraints(lst):
    return list(filter(lambda x: isinstance(x, FunctionalDependency), lst))

def get_domain_constraints(lst):
    return list(filter(lambda x: isinstance(x, DomainConstraint), lst))

def rename_cols_in_constraint(c, name_map):
    if isinstance(c, UniqueConstraint):
        return UniqueConstraint([c1 if c1 not in name_map else 
        name_map[c1] for c1 in c.columns])
    elif isinstance(c, FunctionalDependency):
        dep_column = name_map[c.dependent_column] if c.dependent_column in name_map else c.dependent_column
        other_columns = [name_map[c1] if c1 in name_map else c1 for c1 in c.other_columns]
        expr = get_filter_replacing_field(Field(Expr(c.expr)), name_map).col.expr
        return FunctionalDependency(dep_column, other_columns, expr)
    elif isinstance(c, DomainConstraint):
        constraints = c.constraints
        for o, n in name_map.items():
            if o in c.constraints.keys():
                v = constraints.pop(o)
                constraints[n] = v
        return DomainConstraint(constraints) 
    else:
        return c
def get_cols_used_in_constraint(c):
    if isinstance(c, UniqueConstraint):
        return c.columns
    elif isinstance(c, FunctionalDependency):
        return [c.dependent_column]+c.other_columns
    elif isinstance(c, DomainConstraint):
        return list(c.constraints.keys())

def get_constraint(op):
    if isinstance(op, InitTable):
        op.constraints = get_constraints_from_init_table(op.df)
    elif isinstance(op, InnerJoin):
        renames_left = {}
        renames_right = {}
        op.constraints = []
        for col_name,col_type in op.input_schema_left.items():
            if any([col_name1==col_name for col_name1 in op.input_schema_right]):
                renames_left[col_name] = '{}_x'.format(col_name)
                renames_right[col_name] = '{}_y'.format(col_name)
        if any([isinstance(c, UniqueConstraint) and len(c.columns)==len(op.merge_cols_right) and all([col in c.columns for col in op.merge_cols_right]) for c in op.dependent_ops[1].constraints]):
            op.constraints += [rename_cols_in_constraint(constraint, renames_left) for constraint in get_unique_constraints(op.dependent_ops[0].constraints)+get_fd_constraints(op.dependent_ops[0].constraints)+get_domain_constraints(op.dependent_ops[0].constraints)]
        elif any([isinstance(c, UniqueConstraint) and len(c.columns)==len(op.merge_cols_left) and all([col in c.columns for col in op.merge_cols_left]) for c in op.dependent_ops[0].constraints]):
            op.constraints += [rename_cols_in_constraint(constraint, renames_right) for constraint in get_unique_constraints(op.dependent_ops[1].constraints)+get_fd_constraints(op.dependent_ops[1].constraints)+get_domain_constraints(op.dependent_ops[1].constraints)]
    elif isinstance(op, LeftOuterJoin):
        renames_left = {}
        op.constraints = []
        for col_name,col_type in op.input_schema_left.items():
            if any([col_name1==col_name for col_name1 in op.input_schema_right]):
                renames_left[col_name] = '{}_x'.format(col_name)
        if any([isinstance(c, UniqueConstraint) and len(c.columns)==len(op.merge_cols_right) and all([col in c.columns for col in op.merge_cols_right]) for c in op.dependent_ops[1].constraints]):
            op.constraints += [rename_cols_in_constraint(constraint, renames_left) for constraint in get_unique_constraints(op.dependent_ops[0].constraints)+get_fd_constraints(op.dependent_ops[0].constraints)+get_domain_constraints(op.dependent_ops[0].constraints)]
    elif isinstance(op, GroupBy):
        op.constraints = [UniqueConstraint(op.groupby_cols)]
    elif isinstance(op, DropDuplicate):
        op.constraints = [c for c in op.dependent_ops[0].constraints]
        op.constraints.append(UniqueConstraint(op.cols))
    elif isinstance(op, Pivot): 
        op.constraints = [UniqueConstraint(op.index_col)]
    elif isinstance(op, SetItem):
        op.constraints = [c for c in op.dependent_ops[0].constraints]
        op.constraints.append(FunctionalDependency(op.new_col, get_columns_used(Field(Expr(op.apply_func))), op.apply_func))
    elif isinstance(op, SortValues):
        op.constraints = [c for c in op.dependent_ops[0].constraints]
    elif isinstance(op, DropNA): 
        op.constraints = [c for c in op.dependent_ops[0].constraints]
    elif isinstance(op, FillNA): 
        op.constraints = [c for c in op.dependent_ops[0].constraints]
    elif isinstance(op, Rename):
        op.constraints = [rename_cols_in_constraint(c, op.name_map) for c in op.dependent_ops[0].constraints]
    elif isinstance(op, Copy):
        op.constraints = [c for c in op.dependent_ops[0].constraints]
    elif isinstance(op, DropColumns):
        op.constraints = list(filter(lambda c: not any([c1 in op.cols for c1 in get_cols_used_in_constraint(c)]), op.dependent_ops[0].constraints))
    elif isinstance(op, ChangeType): 
        op.constraints = [c for c in op.dependent_ops[0].constraints]
    elif isinstance(op, GetDummies):
        op.constraints = []
    elif isinstance(op, Append): 
        op.constraints = []
    elif isinstance(op, ConcatColumn):
        op.constraints = []
        for dep_op in op.dependent_ops:
            op.constraints += [c for c in dep_op.constraints]
    elif isinstance(op, UnPivot):
        op.constraints = []
    elif isinstance(op, Split):
        op.constraints = list(filter(lambda c: not any([c1==op.column_to_split for c1 in get_cols_used_in_constraint(c)]), op.dependent_ops[0].constraints))
    elif isinstance(op, Filter):  
        op.constraints = [c for c in op.dependent_ops[0].constraints]
    elif isinstance(op, CrosstableUDF):
        op.constraints = []
    elif isinstance(op, CogroupedMap):
        op.constraints = []

def check_output_filter_rewrite(table_schema, filter1, filter2, constraints):
    symb_table = generate_symbolic_table('table1', table_schema, 2)
    vs = []
    for t in symb_table:
        vs.extend([v1.v for k1,v1 in t.values.items()])
    assumptions = []
    print("table schema = {}".format(table_schema))
    for c in constraints:
        assumptions.append(c.eval(symb_table))
    symb_filter1 = [filter1.eval(tup) for tup in symb_table]
    symb_filter2 = [filter2.eval(tup) for tup in symb_table]
    filter_eq = z3.And(*[symb_filter1[i]==symb_filter2[i] for i in range(len(symb_table))])
    print("constraint: {}".format(z3.simplify(z3.And(*assumptions))))
    print("filter_eq = {}".format(z3.simplify(filter_eq)))
    return check_always_hold(z3.Implies(z3.And(*assumptions), filter_eq))

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
