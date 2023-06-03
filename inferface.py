
from ast import expr

from numpy import sort
from predicate import *
from heapq import merge
import sys
#sys.path.append("./test/")
import z3
import dis
#from test_helper import *
from util import *
from util_type import *
from predicate import *
# from constraint import *
from pandas_op import *
import pickle
from collections import *

t_cnt = 0
def get_some_table_id():
    global t_cnt
    t_cnt += 1
    return t_cnt

def get_schema_from_df(df):
    x = {}
    for col in df.columns:
        if 'object' in str(df[col].dtype) or col == 'n_name':
            x[str(col)] = 'str'
        else:
            x[str(col)] = 'int'
    return x

class Operator(object):
    def __init__(self):
        pass

class InitTable(object):
    def __init__(self, datafile_path, type = '.pickle', skip_read=False):
        self.dependent_ops = []
        self.datafile_path = datafile_path
        if skip_read:
            return
        if datafile_path.endswith('.csv'):
            self.df = pd.read_csv(datafile_path, skipinitialspace=True)
        else:
            self.df = pd.read_pickle(datafile_path)
        self.df.columns = [str(c) for c in self.df.columns]
        self.df['index'] = self.df.index
        self.input_schema = get_schema_from_df(self.df)
    def to_python(self, output_variable_name, input_variable_names={}):
        if self.datafile_path.endswith('.csv'):
            s = "{} = pd.read_csv(\"{}\")\n".format(output_variable_name, self.datafile_path)
        else:
            s = "{} = pickle.load(open(\"{}\",'rb'))\n".format(output_variable_name, self.datafile_path)
        s += '{}.columns = [str(c) for c in {}.columns]\n'.format(output_variable_name, output_variable_name)
        s += '{}["index"] = {}.index'.format(output_variable_name, output_variable_name)
        return s
    def get_inference_instance(self, output_filter):
        input_table = generate_symbolic_table('t{}'.format(get_some_table_id()), self.input_schema, 1)
        self.inference = InitTableInference(input_table, output_filter)
        return self.inference
        

class Filter(Operator):
    def __init__(self, dependent_op, condition):
        self.dependent_ops = [dependent_op]
        self.condition = condition
        # Yin
        self.input_schema = {}
    def get_inference_instance(self, output_filter):
        input_table = generate_symbolic_table('t{}'.format(get_some_table_id()), self.input_schema, 1)
        self.inference = FilterInference(input_table, self.condition, output_filter)
        return self.inference
    def to_python(self, output_variable_name, input_variable_names={}):
        df_var = input_variable_names[self.dependent_ops[0]]
        return "{} = {}[{}]".format(output_variable_name, df_var, pred_to_python(self.condition, df_var))
class InnerJoin(Operator):
    def __init__(self, dependent_op_left, dependent_op_right, cols_left, cols_right):
        self.dependent_ops = [dependent_op_left, dependent_op_right]
        self.merge_cols_left = cols_left if type(cols_left) is list else [cols_left]
        self.merge_cols_right = cols_right if type(cols_right) is list else [cols_right]
        self.input_schema_left = {}
        self.input_schema_right = {}
    def get_inference_instance(self, output_filter):
        table_left = generate_symbolic_table('t{}_l'.format(get_some_table_id()), self.input_schema_left, 1)
        table_right = generate_symbolic_table('t{}_r'.format(get_some_table_id()), self.input_schema_right, 1)
        self.inference = InnerJoinInference(table_left, table_right, self.merge_cols_left, self.merge_cols_right, output_filter)
        return self.inference
    def to_python(self, output_variable_name, input_variable_names={}):
        if len(self.merge_cols_left)==1 and self.merge_cols_left[0] == 'index' and len(self.merge_cols_right)==1 and self.merge_cols_right[0] == 'index':
            return "{} = pd.concat([{},{}], axis=1)".format(output_variable_name, input_variable_names[self.dependent_ops[0]], input_variable_names[self.dependent_ops[1]]) 
        return "{} = {}.merge({}, left_on = [{}], right_on = [{}])".format(output_variable_name, \
                input_variable_names[self.dependent_ops[0]], input_variable_names[self.dependent_ops[1]], \
                ','.join(['"{}"'.format(c) for c in self.merge_cols_left]), ",".join(['"{}"'.format(c) for c in self.merge_cols_right]))
class LeftOuterJoin(Operator):
    def __init__(self, dependent_op_left, dependent_op_right, cols_left, cols_right):
        self.dependent_ops = [dependent_op_left, dependent_op_right]
        self.merge_cols_left = cols_left if type(cols_left) is list else [cols_left]
        self.merge_cols_right = cols_right if type(cols_right) is list else [cols_right]
        self.input_schema_left = {}
        self.input_schema_right = {}
    def get_inference_instance(self, output_filter):
        table_left = generate_symbolic_table('t{}_l'.format(get_some_table_id()), self.input_schema_left, 1)
        table_right = generate_symbolic_table('t{}_r'.format(get_some_table_id()), self.input_schema_right, 2)
        self.inference = LeftOuterJoinInference(table_left, table_right, self.merge_cols_left, self.merge_cols_right, output_filter)
        return self.inference     
    def to_python(self, output_variable_name, input_variable_names={}):
        return "{} = {}.merge({}, how='left', left_on = [{}], right_on = [{}])".format(output_variable_name, \
                input_variable_names[self.dependent_ops[0]], input_variable_names[self.dependent_ops[1]], \
                ','.join(['"{}"'.format(c) for c in self.merge_cols_left]), ",".join(['"{}"'.format(c) for c in self.merge_cols_right]))
class GroupBy(Operator):
    def __init__(self, dependent_op, groupby_cols, aggr_func_map, new_column_names):
        self.dependent_ops = [dependent_op]
        self.groupby_cols = groupby_cols
        self.aggr_func_map = aggr_func_map
        self.new_column_names = new_column_names
        self.input_schema = {}
    def get_inference_instance(self, output_filter):
        input_table = generate_symbolic_table('t{}'.format(get_some_table_id()), self.input_schema, 2)
        self.inference = GroupByInference(input_table, self.groupby_cols, self.aggr_func_map, self.new_column_names, output_filter)
        return self.inference
    def to_python(self, output_variable_name, input_variable_names={}):
        return "{} = {}.groupby([{}]).agg({{ {} }}).reset_index().rename(columns={{ {} }})".format(output_variable_name, input_variable_names[self.dependent_ops[0]], \
            ','.join(['"{}"'.format(c) for c in self.groupby_cols]), \
            ','.join(['"{}":"{}"'.format(k, v[1] if type(v) is tuple else v) for k,v in self.aggr_func_map.items()]), \
            ','.join(['"{}":"{}"'.format(k,v) for k,v in self.new_column_names.items()]))        
class DropDuplicate(Operator):
    def __init__(self, dependent_op, cols):
        self.dependent_ops = [dependent_op]
        self.cols = cols
        # Yin
        self.input_schema = {}
    def get_inference_instance(self, output_filter):
        input_table = generate_symbolic_table('t{}'.format(get_some_table_id()), self.input_schema, 2)
        self.inference = DropDuplicateInference(input_table, self.cols, output_filter)
        return self.inference
    def to_python(self, output_variable_name, input_variable_names={}):
        return "{} = {}.drop_duplicates(subset=[{}])".format(output_variable_name, input_variable_names[self.dependent_ops[0]],\
            ','.join(['"{}"'.format(c) for c in self.cols]))
class Pivot(Operator):
    def __init__(self, dependent_op, index_col, header_col, value_col, value_aggr_func, output_schema):
        self.dependent_ops = [dependent_op]
        self.index_col = index_col if type(index_col) is list else [index_col]
        self.header_col = header_col #if type(header_col) is list else [header_col]
        self.value_col = value_col
        self.value_aggr_func = value_aggr_func
        self.input_schema = {}
        self.output_schema = output_schema
        super().__init__()
    def get_inference_instance(self, output_filter):
        input_table = generate_symbolic_table('t{}'.format(get_some_table_id()), self.input_schema, 2) # TODO: the size of output schema
        self.inference = PivotInference(input_table, self.index_col, self.header_col, self.value_col, self.value_aggr_func, output_filter, self.output_schema)
        return self.inference
    def to_python(self, output_variable_name, input_variable_names={}):
        if self.value_aggr_func is None:
            return '{} = {}.pivot(index=[{}], columns="{}", values="{}").reset_index()'.format(output_variable_name, input_variable_names[self.dependent_ops[0]], \
            ','.join(['"{}"'.format(col) for col in self.index_col]), self.header_col, self.value_col)
        else:
            return '{} = pd.pivot_table({}, index=[{}], columns="{}", values="{}",aggfunc="{}").reset_index()'.format(output_variable_name, input_variable_names[self.dependent_ops[0]], \
            ','.join(['"{}"'.format(col) for col in self.index_col]), self.header_col, self.value_col, self.value_aggr_func)
        
class SetItem(Operator):
    def __init__(self, dependent_op, new_col, apply_func, return_type=None):
        self.dependent_ops = [dependent_op]
        self.new_col = new_col
        self.apply_func = apply_func
        self.return_type = return_type
        self.input_schema = {}
        super().__init__()
    def get_inference_instance(self, output_filter):
        input_table = generate_symbolic_table('t{}'.format(get_some_table_id()), self.input_schema, 1)
        self.inference = SetItemInference(input_table, self.new_col, self.apply_func, self.return_type, output_filter)
        return self.inference
    def to_python(self, output_variable_name, input_variable_names={}):
        lambda_str = get_lambda_code(self.apply_func)
        return '{} = {}\n{}["{}"] = {}.apply({}, axis=1)'.format(output_variable_name, input_variable_names[self.dependent_ops[0]], \
            output_variable_name, self.new_col, output_variable_name, lambda_str)
class SortValues(Operator):
    def __init__(self, dependent_op, cols):
        self.dependent_ops = [dependent_op]
        self.cols = cols if type(cols) is list else [cols]
        self.input_schema = {}
    def get_inference_instance(self, output_filter):
        input_table = generate_symbolic_table('t{}'.format(get_some_table_id()), self.input_schema, 1)
        self.inference = SortValuesInference(input_table, self.cols, output_filter)
        return self.inference
    def to_python(self, output_variable_name, input_variable_names={}):
        return "{} = {}.sort_values(by=[{}])".format(output_variable_name, input_variable_names[self.dependent_ops[0]], \
            ','.join(['"{}"'.format(c) for c in self.cols]))
class DropNA(Operator):
    def __init__(self, dependent_op, cols):
        self.dependent_ops = [dependent_op]
        self.cols = cols if type(cols) is list else [cols]
        self.input_schema = {}
        self.condition = AllAnd(*[IsNotNULL(Field(col)) for col in cols])
    def get_inference_instance(self, output_filter):
        input_table = generate_symbolic_table('t{}'.format(get_some_table_id()), self.input_schema, 1)
        self.inference = DropNAInference(input_table, self.cols, output_filter)
        return self.inference
    def to_python(self, output_variable_name, input_variable_names={}):
        subset_str = ",".join(['"{}"'.format(c) for c in self.cols])
        return "{} = {}.dropna(subset=[{}])".format(output_variable_name, input_variable_names[self.dependent_ops[0]], \
            ','.join(['"{}"'.format(c) for c in self.cols]))
class FillNA(Operator):
    def __init__(self, dependent_op, col, fill_value):
        self.dependent_ops = [dependent_op]
        self.col = col
        self.fill_value = fill_value
        self.input_schema = {}
    def get_inference_instance(self, output_filter):
        input_table = generate_symbolic_table('t{}'.format(get_some_table_id()), self.input_schema, 1)
        self.inference = FillNAInference(input_table, self.col, self.fill_value, output_filter)
        return self.inference
    def to_python(self, output_variable_name, input_variable_names={}):
        return "{}={}\n{}[\"{}\"] = {}[\"{}\"].fillna(value={})".format(output_variable_name, input_variable_names[self.dependent_ops[0]], \
            output_variable_name, self.col, output_variable_name, self.col, const_to_code(self.fill_value))
class Rename(Operator):
    def __init__(self, dependent_op, name_map):
        self.dependent_ops = [dependent_op]
        self.name_map = name_map
        self.input_schema = {}
    def get_inference_instance(self, output_filter):
        input_table = generate_symbolic_table('t{}'.format(get_some_table_id()), self.input_schema, 1)
        self.inference = RenameInference(input_table, self.name_map, output_filter)
        return self.inference
    def to_python(self, output_variable_name, input_variable_names={}):
        if any([v == 'index' for k,v in self.name_map.items()]):
            temp = {}
            index_to_set = ""
            for k,v in self.name_map.items():
                if v != 'index':
                    temp[k] = v
                else:
                    index_to_set = k
            reset_index = "{}['index'] = {}['{}']".format(input_variable_names[self.dependent_ops[0]], input_variable_names[self.dependent_ops[0]], index_to_set)
            if len(temp) == 0:
                return reset_index + "\n{}={}.drop(columns=['{}'])".format(output_variable_name, input_variable_names[self.dependent_ops[0]], index_to_set)
            else:
                return reset_index + "\n{} = {}.drop(columns=['{}']).rename(columns={{ {} }})".format(output_variable_name, input_variable_names[self.dependent_ops[0]], index_to_set, \
            ','.join(['"{}":"{}"'.format(k,v) for k,v in temp.items()]))
        else:
            return "{}={}.rename(columns={{ {} }})".format(output_variable_name, input_variable_names[self.dependent_ops[0]], \
            ','.join(['"{}":"{}"'.format(k,v) for k,v in self.name_map.items()]))
class Copy(Operator):
    def __init__(self, dependent_op):
        self.dependent_ops = [dependent_op]
        self.input_schema = {}
    def get_inference_instance(self, output_filter):
        input_table = generate_symbolic_table('t{}'.format(get_some_table_id()), self.input_schema, 1)
        self.inference = CopyInference(input_table, output_filter)
        return self.inference
    def to_python(self, output_variable_name, input_variable_names={}):
        return "{} = {}.copy()".format(output_variable_name, input_variable_names[self.dependent_ops[0]])
class DropColumns(Operator):
    def __init__(self, dependent_op, cols):
        self.dependent_ops = [dependent_op]
        self.cols = cols
        self.input_schema = {}
    def get_inference_instance(self, output_filter):
        input_table = generate_symbolic_table('t{}'.format(get_some_table_id()), self.input_schema, 1)
        self.inference = DropColumnsInference(input_table, self.cols, output_filter)
        return self.inference
    def to_python(self, output_variable_name, input_variable_names={}):
        return "{} = {}.drop(columns=[{}])".format(output_variable_name, input_variable_names[self.dependent_ops[0]], \
            ','.join(['"{}"'.format(c) for c in self.cols]))
class ChangeType(Operator):
    def __init__(self, dependent_op, target_type, orig_col, new_col):
        self.dependent_ops = [dependent_op]
        self.target_type = target_type
        self.orig_col = orig_col
        self.new_col = new_col
        self.input_schema = {}
    def get_inference_instance(self, output_filter):
        input_table = generate_symbolic_table('t{}'.format(get_some_table_id()), self.input_schema, 1)
        self.inference = ChangeTypeInference(input_table, self.target_type, self.orig_col, self.new_col , output_filter)
        return self.inference
    def to_python(self, output_variable_name, input_variable_names={}):
        if 'datetime' in self.target_type:
            return "{} = {}\n{}['{}'] = pd.to_datetime({}['{}'])".format(output_variable_name, input_variable_names[self.dependent_ops[0]],\
                output_variable_name, self.new_col, output_variable_name, self.orig_col)
        else:
            return "{} = {}\n{}['{}'] = {}['{}'].astype('{}')".format(output_variable_name, input_variable_names[self.dependent_ops[0]],\
                output_variable_name, self.new_col, output_variable_name, self.orig_col, self.target_type)
class Append(Operator): # equivalent to pandas concat with axis=0; append multiple tables vertically, a list of tables
    def __init__(self, dependent_ops):
        self.dependent_ops = dependent_ops
        self.input_schemas = []
    def get_inference_instance(self, output_filter):
        input_tables = []
        for i in range(len(self.input_schemas)):
            input_tables.append(generate_symbolic_table('t{}_{}'.format(get_some_table_id(), i), self.input_schemas[i], 1))
        self.inference = AppendInference(input_tables, output_filter)
        return self.inference
    def to_python(self, output_variable_name, input_variable_names={}):
        return "{} = pd.concat([{}], axis=0)".format(output_variable_name, ','.join([str(input_variable_names[v]) for v in self.dependent_ops]))
class ConcatColumn(Operator): # equivalent to pandas concat with axis=1
    def __init__(self, dependent_ops):
        self.dependent_ops = dependent_ops
        self.input_schemas = []
    def get_inference_instance(self, output_filter):
        input_tables = []
        for i in range(len(self.input_schemas)):
            input_tables.append(generate_symbolic_table('t{}_{}'.format(get_some_table_id(), i), self.input_schemas[i], 1))
        self.inference = ConcatColumnInference(input_tables, output_filter)
        return self.inference
    def to_python(self, output_variable_name, input_variable_names={}):
        return "{} = pd.concat([{}], axis=1)".format(output_variable_name, \
            ','.join([str(input_variable_names[v]) if i==0 else "{}.drop(columns=['index'])".format(input_variable_names[v]) for i,v in enumerate(self.dependent_ops)]))
class UnPivot(Operator):
    def __init__(self, dependent_op, id_vars, value_vars, var_name='variable', value_name='value'): # equivalent to pandas melt
        self.dependent_ops = [dependent_op]
        self.id_vars = id_vars
        self.value_vars = value_vars
        self.var_name = var_name
        self.value_name = value_name
        self.input_schema = {}
    def get_inference_instance(self, output_filter):
        input_tables = []
        input_table = generate_symbolic_table('t{}'.format(get_some_table_id()), self.input_schema, 1)
        self.inference = UnpivotInference(input_table, self.id_vars, self.value_vars, self.var_name, self.value_name, output_filter)
        return self.inference
    def to_python(self, output_variable_name, input_variable_names={}):
        return "{} = {}.melt(id_vars=[{}], value_vars=[{}],var_name=\"{}\", value_name=\"{}\")".format(output_variable_name, input_variable_names[self.dependent_ops[0]],\
            ','.join(['"{}"'.format(c) for c in self.id_vars]),  ','.join(['"{}"'.format(c) for c in self.value_vars]), self.var_name, self.value_name)
class Split(Operator):
    def __init__(self, dependent_op, column_to_split, new_col_names, regex=None, by=None): # split one column into multiple columns, using regex or by
        self.dependent_ops = [dependent_op]
        self.column_to_split = column_to_split
        self.new_col_names = new_col_names
        self.regex = regex
        self.by = by
        self.input_schema = {}
    def get_inference_instance(self, output_filter):
        input_table = generate_symbolic_table('t{}'.format(get_some_table_id()), self.input_schema, 1)
        self.inference = SplitInference(input_table, self.column_to_split, self.new_col_names, self.regex, self.by, output_filter)
        return self.inference
    def to_python(self, output_variable_name, input_variable_names={}):
        if self.regex is None and self.by is not None:
            s = "{} = {}[\"{}\"].str.split('{}',expand=True)\n".format(output_variable_name, input_variable_names[self.dependent_ops[0]], self.column_to_split, self.by)
            s += '{}.columns = [str(x) for x in {}.columns]'.format(output_variable_name, output_variable_name)
            return s
        elif self.regex is not None:
            s = "{} = {}[\"{}\"].str.extract(\"{}\",expand=False)\n".format(output_variable_name, input_variable_names[self.dependent_ops[0]], self.column_to_split, self.regex)
            s += '{}.columns = [str(x) for x in {}.columns]'.format(output_variable_name, output_variable_name)
            return s
        else:
            return ""

class TopN(Operator): # get the top X rows of a dataframe
    def __init__(self, dependent_op, Nrows, sort_order, desc=False):
        self.dependent_ops = [dependent_op]
        self.Nrows = Nrows
        self.sort_order = sort_order if type(sort_order) is list else [sort_order]
        self.input_schema = {}
        self.desc = desc
    def get_inference_instance(self, output_filter):
        input_table = generate_symbolic_table('t{}'.format(get_some_table_id()), self.input_schema, self.Nrows) # TODO: self.Nrows or self.Nrows+1??
        self.inference = TopNInference(input_table, self.Nrows, self.sort_order, self.desc, output_filter)
        return self.inference
    def to_python(self, output_variable_name, input_variable_names={}):
        s = "{} = {}.sort_values(by=[{}], ascending = False)\n".format(output_variable_name, input_variable_names[self.dependent_ops[0]],\
            ','.join(['"{}"'.format(x) for x in self.sort_order]))
        s += "{} = {}.iloc[:{}]".format(output_variable_name, output_variable_name, self.Nrows)
        return s

class GetDummies(Operator):
    def __init__(self, dependent_op, cols, output_schema):
        self.dependent_ops = [dependent_op]
        self.cols = cols if type(cols) is list else [cols]
        self.output_schema = output_schema
        self.input_schema = {}
    def get_inference_instance(self, output_filter):
        input_table = generate_symbolic_table('t{}'.format(get_some_table_id()), self.input_schema, 1)
        self.inference = GetDummiesInference(input_table, self.cols, self.output_schema, output_filter)
        return self.inference
    def to_python(self, output_variable_name, input_variable_names={}):
        return "{} = pd.get_dummies({}, columns = [{}])".format(output_variable_name, input_variable_names[self.dependent_ops[0]], \
            ','.join(['"{}"'.format(c) for c in self.cols]))

class Combine(Operator):
    def __init__(self, dependent_ops, combine_func):
        self.dependent_ops = dependent_ops # two ops
        self.combine_func = combine_func

class CumAggr(Operator): # should use AllAggr interface
    def __init__(self, dependent_op, cum_func): #cum_func in ['max','min','sum','prod']
        self.dependent_ops = [dependent_op]
        self.cum_func = cum_func

class AllAggregate(Operator):
    def __init__(self, dependent_op, initv, aggr_func, convert_to_tuple=lambda x: {'new_aggr_col':x}):
        self.dependent_ops = [dependent_op]
        self.initv = initv
        self.aggr_func = aggr_func
        self.input_schema = {}
        self.convert_to_tuple = convert_to_tuple
        self.return_type = 'str' if type(self.initv.v) is str else 'int'
    def get_inference_instance(self, output_filter):
        input_table = generate_symbolic_table('t{}'.format(get_some_table_id()), self.input_schema, 1)
        self.inference = AllAggrInference(input_table, self.initv, self.aggr_func, output_filter)
        return self.inference
    def to_python(self, output_variable_name, input_variable_names={}):
        s = "{} = functools.reduce({}, [{}] + [row for idx,row in {}.iterrows()])".format(\
                output_variable_name, self.aggr_func, const_to_code(getv(self.initv)), input_variable_names[self.dependent_ops[0]])
        return s


class ScalarComputation(Operator):
    def __init__(self, dependent_op_to_var_map, expr, return_type=None):
        variables = get_lambda_variable_names(expr)
        self.dependent_ops = [dependent_op_to_var_map[v] for v in variables]
        self.dependent_op_to_var_map = dependent_op_to_var_map
        self.expr = expr
        self.input_schema = {}
        self.inference = None
        self.return_type = return_type
    def get_inference_instance(self, output_filter):
        input_tables = []
        if self.return_type is None:
            self.return_type = infer_type_for_expr(self.expr, dependent_var=self.dependent_op_to_var_map)
        self.inference = ScalarComputationInference(input_tables, self.expr, output_filter)
        return self.inference
    def eval(self, tup):
        return self.inference.eval(tup)
    def to_python(self, output_variable_name, input_variable_names={}):
        self.pandas_var_name = output_variable_name
        lambda_vars = get_lambda_variable_names(self.expr)
        return "{} = ({})({})".format(output_variable_name, self.expr, \
            ','.join([input_variable_names[self.dependent_op_to_var_map[v]] for v in lambda_vars]))

class SetItemWithDependency(Operator):
    def __init__(self, df_to_set, new_col, dependent_op_to_var_map, apply_func, return_type='int'):
        assert(any([v==df_to_set for k,v in dependent_op_to_var_map.items()]))
        self.df_to_set = df_to_set
        self.dependent_ops = [v for k,v in dependent_op_to_var_map.items()]
        self.dependent_op_to_var_map = dependent_op_to_var_map
        self.apply_func = apply_func
        self.new_col = new_col
        self.return_type = return_type
    def get_inference_instance(self, output_filter):
        self.inference = SetItemWithDependencyInference([], self.df_to_set, self.new_col, \
            self.dependent_op_to_var_map, self.return_type, self.apply_func, output_filter)
        return self.inference
    def to_python(self, output_variable_name, input_variable_names={}):
        variables = get_lambda_variable_names(self.apply_func)
        df_to_set = ''
        other_vars = []
        other_var_values = []
        for v in variables:
            if isinstance(self.dependent_op_to_var_map[v], ScalarComputation) or isinstance(self.dependent_op_to_var_map[v], AllAggregate):
                other_vars.append(v)
                other_var_values.append(input_variable_names[self.dependent_op_to_var_map[v]])
            else: # df
                df_to_set = v

        lambda_str = self.apply_func[self.apply_func.find(':')+1:]
        new_lambda_str = 'lambda {} : (lambda {} : {})({})'.format(df_to_set, ','.join(other_vars), lambda_str, ','.join(other_var_values))
        return '{} = {}\n{}["{}"] = {}.apply({}, axis=1)'.format(output_variable_name, input_variable_names[self.df_to_set], \
            output_variable_name, self.new_col, output_variable_name, new_lambda_str)

class SubpipeInput(Operator):
    def __init__(self, dependent_op, input_type, group_key=None):
        self.dependent_ops = [dependent_op]
        print("SUBPIPE dependent op = {}".format(type(dependent_op)))
        self.input_type = input_type # table, row, group
        self.input_schema = {}
        if self.input_type == 'group':
            assert(group_key is not None)
        self.group_key = group_key
    def get_inference_instance(self, output_filter):
        if self.input_type in ['table']:
            input_table = generate_symbolic_table('t{}'.format(get_some_table_id()), self.input_schema, 2)
        else:
            input_table = []
        self.inference = SubpipeInputInference(input_table, self.input_type, self.group_key, output_filter)
        return self.inference
    def to_python(self, output_variable_name, input_variable_names={}):
        if self.input_type == 'table':
            return '{} = {}'.format(output_variable_name, input_variable_names[self.dependent_ops[0]])
        elif self.input_type == 'row':
            return '{} = row'.format(output_variable_name)
        elif self.input_type == 'group':
            return '{} = group_df'.format(output_variable_name)


class PipelinePath(object):
    def __init__(self, operators, path_cond=True): # path_cond: ScalarComputation, as the condition should be a scalar boolean value
        self.operators = operators
        self.cond = path_cond
    def to_python(self, output_variable_name, input_variable_names={}):
        lines = []
        temp_input_vars = {k:v for k,v in input_variable_names.items()}
        cond_var = 'cond_{}'.format(output_variable_name)
        for i,op in enumerate(self.operators):
            output_var = 'subpipe_{}_op_{}'.format(output_variable_name, i)
            temp_input_vars[op] = output_var
            lines.append(op.to_python(output_var, temp_input_vars))
        if type(self.cond) is not bool:
            lines.append(self.cond.to_python(cond_var, temp_input_vars))
        return '\n'.join(lines) 

class SubPipeline(object): # output can be scalar/row/df
    def __init__(self, paths):
        if type(paths) is list:
            self.paths = paths
        else:
            self.paths = [paths]
    def to_python_helper(self, input_variable_names={}):
        lines = []
        if len(self.paths) == 1: # no condition
            p = self.paths[0]
            lines.append(p.to_python('0', input_variable_names))
            lines.append('return subpipe_{}_op_{}'.format('0', len(p.operators)-1))
        else:
            for i,p in enumerate(self.paths):
                lines.append(p.to_python(str(i), input_variable_names))
            s = '' # return {} if {} else ({})
            for i,p in enumerate(self.paths):
                if i == 0:
                    s = 'subpipe_{}_op_{}'.format(i, len(p.operators)-1)
                else:
                    s = 'subpipe_{}_op_{} if cond_{} else ({})'.format(i, len(p.operators)-1, i, s)
            lines.append("return {}".format(s))
        return '\n'.join(lines)

class SubpipeUDF(Operator):
    def __init__(self, subpipeline):
        self.dependent_ops = []#set()
        self.subpipeline = subpipeline
        for path in self.subpipeline.paths:
            for op in path.operators:
                if isinstance(op, SubpipeInput):

                    if op not in self.dependent_ops:
                        self.dependent_ops.append(op.dependent_ops[0])
 
class CrosstableUDF(SubpipeUDF):
    
    def __init__(self, df_to_set, new_col, subpipeline, return_type=None):
        self.df_to_set = df_to_set
        self.new_col = new_col
        self.subpipeline = subpipeline
        self.return_type = return_type
        self.input_schema_left = {}
        self.input_schema_right = {}
        super(CrosstableUDF, self).__init__(subpipeline)
    def get_inference_instance(self, output_filter):
        # not consistant
        input_table_left = generate_symbolic_table('t{}'.format(get_some_table_id()), self.input_schema_left, 1) # for dependent_op
        input_table_right = None
        for path in self.subpipeline.paths:
            for op in path.operators:
                if isinstance(op, SubpipeInput) and op.input_type == 'table':
                    input_table_right = generate_symbolic_table('t{}'.format(get_some_table_id()), self.input_schema_right, 1) # for dependent_op
        path_infs = []
        for path in self.subpipeline.paths:
            pipe_full_len = len(path.operators) if type(path.cond) is bool else len(path.operators)+1
            sub_pipeline = [None for i in range(pipe_full_len)]
            sub_pipeline_dependency = [[] for i in range(pipe_full_len)]
            op_id_map = {op:i for i,op in enumerate(path.operators)}
            for i,op in reversed(list(enumerate(path.operators+([] if type(path.cond) is bool else [path.cond])))):
                inference = op.get_inference_instance(None)
                inference.input_filters = [None, None]
                
                sub_pipeline[i] = inference
                sub_pipeline_dependency[i] = [op_id_map[op_] if op_ in op_id_map else [] for op_ in op.dependent_ops]
            
            path_inf = CrosstableUDFOnePathInference(input_table_left, self.new_col, sub_pipeline, sub_pipeline_dependency, \
                True if type(path.cond) is bool else sub_pipeline[-1], output_filter)
            path_infs.append(path_inf)
        self.inference = CrosstableUDFInference(path_infs, output_filter)
        if self.return_type is None:
            self.return_type = self.subpipeline.paths[0].operators[-1].return_type
        return self.inference
    
    def to_python(self, output_variable_name, input_variable_names={}):
        func_body = self.subpipeline.to_python_helper(input_variable_names)
        idx = func_body.split('\n')[0].index('=')
        input_table_name = func_body.split('\n')[0][idx+2:]
        udf_name = "udf_{}".format(hash(self)%10000)
        ret = "def {}(row, {}):\n".format(udf_name, input_table_name)
        ret += '\n'.join(['\t'+line for line in func_body.split('\n')])
        ret += '\n'
        ret += '{} = {}\n{}["{}"] = {}.apply({}, axis=1, args = [{}])'.format(output_variable_name, input_variable_names[self.df_to_set],\
            output_variable_name, self.new_col, output_variable_name, udf_name, input_table_name)
        return ret

class GroupedMap(SubpipeUDF): # group df -> df/scalar
    def __init__(self, subpipeline):
        self.subpipeline = subpipeline
        super(GroupedMap, self).__init__(subpipeline)
    def get_return_type(self):
        last_op = self.subpipeline.paths[0].operators[-1]
        if isinstance(last_op, ScalarComputation) or isinstance(last_op, AllAggregate):
            return 'int'
        else:
            return 'table'
    def get_inference_instance(self, output_filter):
        if not hasattr(self, 'input_schema'):
            for path in self.subpipeline.paths:
                for op in path.operators:
                    if isinstance(op, SubpipeInput):
                        self.input_schema = op.input_schema
        input_table = generate_symbolic_table('t{}'.format(get_some_table_id()), self.input_schema, 2)
        path_infs = []
        group_cols = []
        for path in self.subpipeline.paths:
            pipe_full_len = len(path.operators) if type(path.cond) is bool else len(path.operators)+1
            sub_pipeline = [None for i in range(pipe_full_len)]
            sub_pipeline_dependency = [[] for i in range(pipe_full_len)]
            op_id_map = {op:i for i,op in enumerate(path.operators)}
            for i,op in reversed(list(enumerate(path.operators+([] if type(path.cond) is bool else [path.cond])))):

                inference = op.get_inference_instance(None)
                inference.input_filters = [None, None]
                
                sub_pipeline[i] = inference
                sub_pipeline_dependency[i] = [op_id_map[op_] if op_ in op_id_map else [] for op_ in op.dependent_ops]
            
            path_inf = GroupedMapOnePathInference(input_table, group_cols, sub_pipeline, sub_pipeline_dependency, \
                True if type(path.cond) is bool else sub_pipeline[-1], output_filter)
            path_infs.append(path_inf)
        self.inference = GroupedMapInference(path_infs, output_filter)
        return self.inference
    def get_group_cols(self):
        group_cols = []
        for op in self.subpipeline.paths[0].operators:
            if isinstance(op, SubpipeInput):
                group_cols = op.group_key
        return group_cols
    def to_python(self, output_variable_name, input_variable_names={}):
        func_body = self.subpipeline.to_python_helper(input_variable_names)
        udf_name = "udf_{}".format(hash(self)%10000)
        ret = "def {}(group_df):\n".format(udf_name)
        ret += '\n'.join(['\t'+line for line in func_body.split('\n')])
        ret += '\n'
        group_cols = self.get_group_cols()
        if self.get_return_type() == 'table':
            new_cols = list(self.input_schema_left.keys())
            for c in group_cols:
                new_cols.remove(c)
            ret += '{} = {}.groupby([{}]).apply({})[[{}]].reset_index()'.format(output_variable_name, input_variable_names[self.dependent_ops[0]], \
                ','.join(['"{}"'.format(col) for col in group_cols]), udf_name,\
                ",".join(['"{}"'.format(col) for col in list(filter(lambda x: x!='index', [col for col,dtype in self.input_schema_left.items()]))]))
        else:
            ret += '{} = {}.groupby([{}]).apply({}).reset_index()\n'.format(output_variable_name, input_variable_names[self.dependent_ops[0]], ','.join(['"{}"'.format(col) for col in group_cols]), udf_name)
            ret += "{}.columns = [str(x) for x in {}.columns]".format(output_variable_name, output_variable_name)
        return ret

class CogroupedMap(GroupedMap): # group df, group df -> df
    def __init__(self, subpipeline):
        super(CogroupedMap, self).__init__(subpipeline)
        



def generate_output_schemas(ops):
    output_schemas = {}
    for i,op in enumerate(ops):
        if isinstance(op, InnerJoin) or isinstance(op, LeftOuterJoin): # has more than 1 operators
            op.input_schema_left, op.input_schema_right = output_schemas[op.dependent_ops[0]],  output_schemas[op.dependent_ops[1]]
            output_schemas[op] = infer_output_schema([output_schemas[op.dependent_ops[0]], output_schemas[op.dependent_ops[1]]], op)
        elif isinstance(op, InitTable):
            output_schemas[op] = op.input_schema
        elif isinstance(op, Pivot):
            op.input_schema = output_schemas[op.dependent_ops[0]]
            for col in op.index_col:
                if col not in op.output_schema:
                    op.output_schema[col] = op.input_schema[col]
            output_schemas[op] = op.output_schema
        elif isinstance(op, SubpipeUDF):
            output_schemas[op] = get_schema_by_subpipeline(op, output_schemas)
        elif isinstance(op, GetDummies):
            output_schemas[op] = op.output_schema
        elif isinstance(op, Append) or isinstance(op, ConcatColumn):
            for j in range(len(op.dependent_ops)):
                op.input_schemas.append(output_schemas[op.dependent_ops[j]])
            output_schemas[op] = infer_output_schema(op.input_schemas, op)
        else:
            op.input_schema = output_schemas[op.dependent_ops[0]]
            output_schemas[op] = infer_output_schema(output_schemas[op.dependent_ops[0]], op)
    return output_schemas

def merge_dicts(*dict_args):
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result

def infer_output_schema(input_schemas, op):
    if type(input_schemas) is not list:
        input_schemas = [input_schemas]
    if isinstance(op, Filter):
        return input_schemas[0]
    elif isinstance(op, TopN):
        return input_schemas[0]
    elif isinstance(op, DropDuplicate):
        return input_schemas[0]
    elif isinstance(op, Copy):
        return input_schemas[0]
    elif isinstance(op, Append):
        return merge_dicts(*input_schemas)
        res = input_schemas[0]
        for i in input_schemas[1:]:
            res = dict(res.items() + i.items())    
        return res
    elif isinstance(op, ConcatColumn):
        return merge_dicts(*input_schemas)
        res = input_schemas[0]
        for i in input_schemas[1:]:
            res = dict(res.items() + i.items())    
        return res
    elif isinstance(op, UnPivot):
        x = {k:v for k,v in input_schemas[0].items()}
        # only keep the columns in id_vars
        # remove all other columns
        vtype = 'int'
        for k,v in input_schemas[0].items():
            if k not in op.id_vars:
                t = x.pop(k)
            if k in op.value_vars:
                vtype = v
        x[op.var_name] = 'str'
        x[op.value_name] = vtype
        x['index'] = 'int'
        return x
    elif isinstance(op, DropNA):
        return input_schemas[0]
    elif isinstance(op, FillNA):
        return input_schemas[0]
    elif isinstance(op, SortValues):
        return input_schemas[0]
    elif isinstance(op, TopN):
        return input_schemas[0]
    elif isinstance(op, SubpipeInput):
        return input_schemas[0]
    elif isinstance(op, DropColumns): 
        x = {k:v for k,v in input_schemas[0].items()}
        for c in op.cols:
            x.pop(c)
        return x
    elif isinstance(op, Split): 
        x = {'index':'int'}
        t = 'str'
        for i in op.new_col_names:
            x[i] = t 
        return x
    elif isinstance(op, Rename):
        x = {}
        for k,v in input_schemas[0].items():
            if k in op.name_map:
                x[op.name_map[k]] = v
            elif k not in x:
                x[k] = v
        return x
    elif isinstance(op, ChangeType):
        x = {k:v for k,v in input_schemas[0].items()}
        x[op.new_col] = x.pop(op.orig_col)   # first change name
        x[op.new_col] = convert_type_to_z3_types(op.target_type) # next change type
        return x
    elif isinstance(op, SetItem) or isinstance(op, CrosstableUDF):
        if op.new_col in input_schemas[0]:
            return input_schemas[0]
        else:
            x = {k:v for k,v in input_schemas[0].items()}
            # TODO: infer type from lambda
            if hasattr(op, 'return_type') and op.return_type is not None:
                x[op.new_col] = op.return_type
            else:
                x[op.new_col] = 'int'
            return x
    elif isinstance(op, InnerJoin):
        x = {k:v for k,v in input_schemas[0].items()}
        for k,v in input_schemas[1].items():
            if k in op.merge_cols_left:
                continue
            else:
                if k in x.keys():
                    x[k+"_x"] = x.pop(k)
                    x[k+"_y"] = v
                else:
                    x[k] = v
        return x
    elif isinstance(op, LeftOuterJoin):
        x = {k:v for k,v in input_schemas[0].items()}
        for k,v in input_schemas[1].items():
            if k in op.merge_cols_left:
                continue
            else:
                if k in x.keys():
                    x[k+"_x"] = x.pop(k)
                    x[k+"_y"] = v
                else:
                    x[k] = v
        return x
    elif isinstance(op, GroupBy):
        x = {k:input_schemas[0][k] for k in op.groupby_cols}
        x.update({v:'int' for k,v in op.new_column_names.items()})
        return x
    elif isinstance(op, AllAggregate):
        # TODO: infer type or given type
        temp = op.convert_to_tuple([0 for i in range(100)])
        x = {k:'int' for k,v in temp.items()}
        return x

def get_subpipeline_input_schema(subpipeline, output_schemas):
    for p in subpipeline.paths:
        for k in p.operators:
            if len(k.dependent_ops) == 1:
                k.input_schema = output_schemas[k.dependent_ops[0]]
            else:
                k.input_schema = [output_schemas[k.dependent_ops[0]], output_schemas[k.dependent_ops[1]]]
                k.input_schema_left, k.input_schema_right = output_schemas[k.dependent_ops[0]], output_schemas[k.dependent_ops[1]]
            if isinstance(k, SubpipeInput):
                output_schemas[k] = output_schemas[k.dependent_ops[0]]
            if isinstance(k, AllAggregate) or isinstance(k, ScalarComputation):
                output_schemas[k] =  {'res':'int'}
            else:
                output_schemas[k] = infer_output_schema(k.input_schema, k)
    return output_schemas
            
def get_subpipeline_inputs(subpipeline):
    res = []
    for p in subpipeline.paths:
        for k in p.operators:
            if isinstance(k, SubpipeInput) and k not in res:
                res.append(k)
    return res


def get_schema_by_subpipeline(op, output_schemas):
    sub_pipe = op.subpipeline
    output_schemas = get_subpipeline_input_schema(sub_pipe, output_schemas)
    sub_pipe_inputs = get_subpipeline_inputs(sub_pipe)
    for s in sub_pipe_inputs:
        s.input_schema = output_schemas[s.dependent_ops[0]]
        # handle the case for group
        if s.input_type == 'group':
            if op.get_return_type() == 'table':
                op.input_schema_left = output_schemas[s.dependent_ops[0]]
                x = output_schemas[s.dependent_ops[0]]
            else:
                group_cols = op.get_group_cols()
                x = {col: output_schemas[s.dependent_ops[0]][col] for col in list(filter(lambda x: x in group_cols, [k for k,v in output_schemas[s.dependent_ops[0]].items()]))}
                x['0'] = op.subpipeline.paths[0].operators[-1].return_type
            return x 
        # handle the row - table case:
        elif s.input_type == 'row':
            op.input_schema_left = output_schemas[s.dependent_ops[0]]
        elif s.input_type ==' table':
            op.input_schema_right = output_schemas[s.dependent_ops[1]]
    # print("OUTPUT SCHEMA = {}".format(op.input_schema_left))

    if hasattr(op, 'new_col'):
        return_type = op.return_type if op.return_type is not None else 'int'
        op.input_schema_left[op.new_col] = return_type

    return op.input_schema_left



def generate_output_schemas_sub(sub_input):
    output_schemas_sub = {}
    for i, op in enumerate(sub_input):
        if isinstance(op, SubpipeInput):
            output_schemas_sub[op] = op.dependents[0].input_schema
    return output_schemas_sub

def pred_to_python(expr, df_var):
    if isinstance(expr, And):
        if not isinstance(expr.rh.rh, Constant):
            return pred_to_python(expr.lh, df_var)
        lh = pred_to_python(expr.lh, df_var)
        rh = pred_to_python(expr.rh, df_var)
        return "({} & {})".format(lh, rh)
    elif isinstance(expr, Or):
        lh = pred_to_python(expr.lh, df_var)
        rh = pred_to_python(expr.rh, df_var)
        return "({} | {})".format(lh, rh)
    elif isinstance(expr, BinOp):
        if expr.op == 'in':
            return "({}.isin({}))".format(pred_to_python(expr.lh, df_var), pred_to_python(expr.rh, df_var))
        elif expr.op == 'subset':
            return "({}.str.contains({}))".format(pred_to_python(expr.lh, df_var), pred_to_python(expr.rh, df_var))
        else:
            return "({} {} {})".format(pred_to_python(expr.lh, df_var), expr.op, pred_to_python(expr.rh, df_var))
    elif isinstance(expr, UnaryOp):
        if isinstance(expr, Not):
            return "(~{})".format(pred_to_python(expr.v, df_var))
        elif isinstance(expr, IsNULL):
            return "({}.isnull())".format(pred_to_python(expr.v, df_var))
        elif isinstance(expr, IsNotNULL):
            return "({}.notnull())".format(pred_to_python(expr.v, df_var))
    elif isinstance(expr, Variable):
        return expr.v
    elif isinstance(expr, Constant):
        if 'datetime' in expr.typ:
            return "pd.to_datetime({})".format(const_to_code(expr.v))
        else:
            return const_to_code(expr.v)
    elif isinstance(expr, Field):
        if type(expr.col) is str:
            return "{}['{}']".format(df_var, expr.col)
        elif isinstance(expr.col, Expr):
            var = get_lambda_varibale_name(expr.col.expr)
            return "({})".format(str(expr.col).replace(var,df_var))
    elif isinstance(expr, bool):
        return str(expr)
    elif isinstance(expr, String):
        return const_to_code(expr)
    else:
        print("CANNOT HANDLE {} {}".format(expr, type(expr)))

def pred_to_python_using_lambda(expr, df_var):
    if isinstance(expr, And): 
        lh = pred_to_python_using_lambda(expr.lh, df_var)
        rh = pred_to_python_using_lambda(expr.rh, df_var)
        return "({}) and ({})".format(lh, rh)
    elif isinstance(expr, Or): 
        lh = pred_to_python_using_lambda(expr.lh, df_var)
        rh = pred_to_python_using_lambda(expr.rh, df_var)
        return "({}) or ({})".format(lh, rh)
    elif isinstance(expr, BinOp):
        if isinstance(expr.rh, Constant):
            if is_null_constant(expr.rh.v):
                if expr.op == '==':
                    return "pd.isnull({})".format(pred_to_python_using_lambda(expr.lh, df_var))
                elif expr.op == '!=':
                    return "pd.notnull({})".format(pred_to_python_using_lambda(expr.lh, df_var))
                else:
                    assert(False)
            else:
                return "{} {} {}".format(pred_to_python_using_lambda(expr.lh, df_var), expr.op, pred_to_python_using_lambda(expr.rh, df_var))
        if isinstance(expr.rh, Variable):
            # decompose it to a set of Or constant
            pred_new_lst = []
            for i in expr.rh.values:
                pred_new_lst.append(BinOp(expr.lh, expr.op, Constant(i, expr.rh.typ)))
            pred_new = AllOr(*pred_new_lst)
            return pred_to_python_using_lambda(pred_new, df_var)
        else:
            if expr.op == 'subset':
                return "{} {} {}".format(pred_to_python_using_lambda(expr.lh, df_var), 'in', pred_to_python_using_lambda(expr.rh, df_var))
            else:
                return "{} {} {}".format(pred_to_python_using_lambda(expr.lh, df_var), expr.op, pred_to_python_using_lambda(expr.rh, df_var))
            
    elif isinstance(expr, UnaryOp):
        if isinstance(expr, Not):
            return "(not ({}))".format(pred_to_python_using_lambda(expr.v, df_var))
        elif isinstance(expr, IsNULL):
            return "pd.isnull({})".format(pred_to_python_using_lambda(expr.v, df_var))
        elif isinstance(expr, IsNotNULL):
            return "not pd.isnull({})".format(pred_to_python_using_lambda(expr.v, df_var))
    elif isinstance(expr, Constant):
        if 'datetime' in expr.typ:
            return "pd.to_datetime('{}')".format(const_to_code(expr.v))
        else:
            return const_to_code(expr.v)
    elif isinstance(expr, String):
        return const_to_code(expr)
    elif isinstance(expr, Field):
        if type(expr.col) is str:
            return "row['{}']".format(expr.col)
    elif isinstance(expr, Expr):
        var = get_lambda_varibale_name(expr.expr)
        return "({})".format(str(expr).replace(var,'row'))
    elif isinstance(expr, bool):
        return str(expr)
    elif isinstance(expr, str):
        return expr
    elif isinstance(expr, ScalarComputation): # TODO: temp fix
        return expr.pandas_var_name
    else:
        print("CANNOT HANDLE {} {}".format(expr, type(expr)))

from collections import Counter
def infer_type_for_expr(expr, table_schema={}, dependent_var={}):
    vars = get_lambda_variable_names(expr)
    columns = get_column_used_from_lambda(expr)
    c = Counter()
    for var in vars:
        if (var in dependent_var and isinstance(dependent_var[var], SubpipeInput)) \
            or len(table_schema) > 0: # var is a row
            schema = table_schema if len(table_schema) > 0 else dependent_var[var].input_schema
            for col in columns:
                if col in schema:
                    c[schema[col]] += 1
        elif var in dependent_var and hasattr(dependent_var[var], 'return_type'):
            c[dependent_var[var].return_type] += 1
    if len(c) == 0:
        return 'int'
    return c.most_common()[0]


def determine_superset_simple(op, candidate):
    # false means: not superset, true means might have superset
    cols_used = get_columns_used(candidate)
    if isinstance(op, InnerJoin) or isinstance(op, LeftOuterJoin):
        if set(cols_used).isdisjoint(set(op.merge_cols_left).union(set(op.merge_cols_right))):
            # does not contain predicate on the join col
            return True
    else:
        return False
