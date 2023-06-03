from heapq import merge
from operator import imod
import z3
import dis
from util import *
from util_type import *
from predicate import *
from constraint import *
from pandas_op import *

t_cnt = 0
def get_some_table_id():
    global t_cnt
    t_cnt += 1
    return t_cnt

class Operator(object):
    def __init__(self):
        self.dependent_ops = []
        self.inference = None

class Filter(Operator):
    def __init__(self, input_schema, condition):
        self.input_schema = input_schema
        self.condition = condition
        super().__init__()
    def get_inference_instance(self, output_filter):
        input_table = generate_symbolic_table('t{}'.format(get_some_table_id()), self.input_schema, 1)
        self.inference = FilterInference(input_table, self.condition, output_filter)
        return self.inference
class InnerJoin(Operator):
    def __init__(self, input_schema_left, input_schema_right, merge_col):
        self.input_schema_left = input_schema_left
        self.input_schema_right = input_schema_right
        self.merge_col = merge_col if type(merge_col) is list else [merge_col]
        super().__init__()
    def get_inference_instance(self, output_filter):
        table_left = generate_symbolic_table('t{}_l'.format(get_some_table_id()), self.input_schema_left, 1)
        table_right = generate_symbolic_table('t{}_r'.format(get_some_table_id()), self.input_schema_right, 1)
        self.inference = InnerJoinInference(table_left, table_right, self.merge_col, output_filter)
        return self.inference
class GroupBy(Operator):
    def __init__(self, input_schema, groupby_cols, aggr_func_map, new_column_names):
        self.input_schema = input_schema
        self.groupby_cols = groupby_cols
        self.aggr_func_map = aggr_func_map
        self.new_column_names = new_column_names
        super().__init__()
    def get_inference_instance(self, output_filter):
        input_table = generate_symbolic_table('t{}'.format(get_some_table_id()), self.input_schema, 2)
        self.inference = GroupByInference(input_table, self.groupby_cols, self.aggr_func_map, self.new_column_names, output_filter)
        return self.inference
class DropDuplicate(Operator):
    def __init__(self, input_schema, cols):
        self.input_schema = input_schema
        self.cols = cols
        super().__init__()
    def get_inference_instance(self, output_filter):
        input_table = generate_symbolic_table('t{}'.format(get_some_table_id()), self.input_schema, 2)
        self.inference  = DropDuplicateInference(input_table, self.cols, output_filter)
        return self.inference
class Pivot(Operator):
    def __init__(self, input_schema, index_col, header_col, value_col, value_aggr_func, output_schema):
        self.input_schema = input_schema
        self.index_col = index_col if type(index_col) is list else [index_col]
        self.header_col = header_col if type(header_col) is list else [header_col]
        self.value_col = value_col
        self.value_aggr_func = value_aggr_func
        self.output_schema = output_schema
        super().__init__()
    def get_inference_instance(self, output_filter):
        input_table = generate_symbolic_table('t{}'.format(get_some_table_id()), self.input_schema, 2) # TODO: the size of output schema
        self.inference = PivotInference(input_table, self.index_col, self.header_col, self.value_col, self.value_aggr_func, output_filter, self.output_schema)
        return self.inference
class SetItem(Operator):
    def __init__(self, input_schema, new_col, apply_func):
        self.input_schema = input_schema
        self.new_col = new_col
        self.apply_func = apply_func
        super().__init__()
    def get_inference_instance(self, output_filter):
        input_table = generate_symbolic_table('t{}'.format(get_some_table_id()), self.input_schema, 1)
        self.inference = SetItemInference(input_table, self.new_col, self.apply_func, output_filter)
        return self.inference
class SortValues(Operator):
    def __init__(self, input_schema, cols):
        self.input_schema = input_schema
        self.cols = cols if type(cols) is list else [cols]
        super().__init__()
    def get_inference_instance(self, output_filter):
        input_table = generate_symbolic_table('t{}'.format(get_some_table_id()), self.input_schema, 1)
        self.inference = SortValuesInference(input_table, self.cols, output_filter)
        return self.inference
class DropNA(Operator):
    def __init__(self, input_schema, cols):
        self.input_schema = input_schema
        self.cols = cols if type(cols) is list else [cols]
        super().__init__()
    def get_inference_instance(self, output_filter):
        input_table = generate_symbolic_table('t{}'.format(get_some_table_id()), self.input_schema, 1)
        self.inference = DropNAInference(input_table, self.cols, output_filter)
        return self.inference
class FillNA(Operator):
    def __init__(self, input_schema, col, fill_value):
        self.input_schema = input_schema
        self.col = col
        self.fill_value = fill_value
        super().__init__()
    def get_inference_instance(self, output_filter):
        input_table = generate_symbolic_table('t{}'.format(get_some_table_id()), self.input_schema, 1)
        self.inference = FillNAInference(input_table, self.col, self.fill_value, output_filter)
        return self.inference
class Rename(Operator):
    def __init__(self, input_schema, name_map):
        self.input_schema = input_schema
        self.name_map = name_map
        super().__init__()
    def get_inference_instance(self, output_filter):
        input_table = generate_symbolic_table('t{}'.format(get_some_table_id()), self.input_schema, 1)
        self.inference = RenameInference(input_table, self.name_map, output_filter)
        return self.inference
class AllAggregate(Operator):
    def __init__(self, input_schema, initv, aggr_func, convert_to_tuple):
        self.input_schema = input_schema
        self.initv = initv
        self.aggr_func = aggr_func
        self.convert_to_tuple = convert_to_tuple
    def get_inference_instance(self, output_filter):
        input_table = generate_symbolic_table('t{}'.format(get_some_table_id()), self.input_schema, 1)
        self.inference = AllAggrInference(input_table, self.initv, self.aggr_func, output_filter, self.convert_to_tuple)
        return self.inference


def infer_output_schema(input_schemas, op):
    if type(input_schemas) is not list:
        input_schemas = [input_schemas]
    if isinstance(op, Filter):
        return input_schemas[0]
    elif isinstance(op, DropDuplicate):
        return input_schemas[0]
    elif isinstance(op, Rename):
        x = {}
        for k,v in input_schemas[0].items():
            if k in op.name_map:
                x[op.name_map[k]] = v
            else:
                x[k] = v
        return x
    elif isinstance(op, SetItem):
        if op.new_col in input_schemas[0]:
            return input_schemas[0]
        else:
            x = {k:v for k,v in input_schemas[0].items()}
            # TODO: infer type from lambda
            x[op.new_col] = 'int'
            return x
    elif isinstance(op, InnerJoin):
        x = {k:v for k,v in input_schemas[0].items()}
        for k,v in input_schemas[1].items():
            if k == op.merge_col:
                continue
            x[k] = v
        return x
    elif isinstance(op, GroupBy):
        x = {k:input_schemas[0][k] for k in op.groupby_cols}
        # TODO: infer type of lambda
        x.update({v:'int' for k,v in op.new_column_names.items()})
        return x
    elif isinstance(op, AllAggregate):
        # TODO: infer type or given type
        temp = op.convert_to_tuple([0 for i in range(100)])
        x = {k:'int' for k,v in temp.items()}
        return x
    