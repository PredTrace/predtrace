import pandas
import z3
import dis
from util import *
from predicate import *
# from constraint import *
from pandas_op import *
#from infer_schema import *
from interface import *
from table_constraint import *
import copy
import psycopg2
import time

def get_previous_op(ops, op):
    res = []
    for o in ops:
        if op in o.dependent_ops:
            res.append(o)
    return res


def check_operator_in_cond(expr):
    # check if there is a operator
    # print(expr)
    if isinstance(expr, BoolOp):
        return check_operator_in_cond(expr.lh) or check_operator_in_cond(expr.rh)
    if isinstance(expr, UnaryOp):
        return check_operator_in_cond(expr.v)
    if isinstance(expr, BinOp):
        if hasattr(expr, 'rh') and isinstance(expr.rh, ScalarComputation):
            return True
        else:
            return False


def get_cond_with_variable(expr):
    expr_new = copy.copy(expr)
    if isinstance(expr_new, BoolOp):
        if check_operator_in_cond(expr_new.lh):
            expr_new.lh = get_cond_with_variable(expr_new.lh)
        if check_operator_in_cond(expr_new.rh):
            expr_new.rh = get_cond_with_variable(expr_new.rh)
        return expr_new
    if isinstance(expr_new, UnaryOp):
        expr_new.v = get_cond_with_variable(expr_new.v)
        return expr_new
    if isinstance(expr_new, BinOp):
        if hasattr(expr_new, 'rh') and isinstance(expr_new.rh, ScalarComputation):
            typ = expr_new.rh.return_type
            col = get_column_used_from_lambda(expr_new.rh.expr)
            if typ[0] == 'int':
                expr_new.rh = Variable(z3.Int('some-v'), col[0], typ = 'int')
            elif typ[0] == 'str':
                expr_new.rh = Variable(z3.String('some-v'), col[0], typ = 'str')
        return expr_new
    else:
        return expr_new

        

def generate_output_filter_from_previous(op, ops):
    previous_ops = get_previous_op(ops, op)
    if len(previous_ops) == 0: # the last operator in the data pipeline.
        return {None:None}
    else:
        output_filters = {}
        for p_op in previous_ops:
            idx = p_op.dependent_ops.index(op)
            output_filters[p_op] = p_op.inference.input_filters[idx]
        return output_filters

        
def get_updated_cols(op, output_filter, input_schema, output_schemas):
    # two special condition: groupby, changetye
    columns_used = get_columns_used(output_filter)
    input_cols = set(input_schema.keys())
    res = set()
    for k,v in output_schemas[op].items():
        if k not in input_cols:
            res.add(k)
        else:
            if v != input_schema[k]:
                res.add(k)
    # for groupby
    if hasattr(op, 'new_column_names'):
        agg_col = set(op.new_column_names.values())
        res = res.union(agg_col)
    # for dropduplicate
    elif isinstance(op, DropDuplicate):
        other_cols = set(columns_used) - set(op.cols)
        res = res.union(other_cols)
    res = res.intersection(columns_used)
    return res


def check_rewrite_rules(op):
    if hasattr(op, 'name_map') or  hasattr(op, 'target_type') or hasattr(op, 'apply_func') \
        or hasattr(op, 'fill_value') or hasattr(op, 'new_col_names') or hasattr(op, 'output_schema') or hasattr(op, 'value_name'):
        return False
    else:
        return True


def rewrite_path(paths):
    sub_ops = paths.operators
    left_path =[]
    right_path = []
    for o in sub_ops:
        if isinstance(o, SubpipeInput):
            if o.input_type == 'row':
                left_path.append(o)
            elif o.input_type == 'table':
                right_path.append(o)
            continue
        for d in o.dependent_ops:
            if d in left_path:
                left_path.append(o)
                break
            elif d in right_path:
                right_path.append(o)
                break
    return  left_path, right_path



def get_pred_on_new_col(expr, new_col):
    if isinstance(expr, BoolOp):
        if expr_contain_column(expr.rh, new_col):
            return get_pred_on_new_col(expr.rh, new_col)
        elif expr_contain_column(expr.lh, new_col):
            return get_pred_on_new_col(expr.lh, new_col)
        else:
            return True
    elif isinstance(expr, BinOp):
        if expr.lh.col == new_col:
            return expr
        else:
            return True

def get_from_left(expr, col):
    if isinstance(expr, BoolOp):
        if get_from_left(expr.lh, col) is not None:
            return get_from_left(expr.lh, col)
        else:
            return get_from_left(expr.rh, col)
    elif isinstance(expr, BinOp):
        if expr.lh.col == col:
            return expr
        else:
            pass
    elif isinstance(expr, UnaryOp):
        return get_from_left(expr.v, col)
    else:
        pass    

def get_pred_used(pred_left, pred_right):
    if isinstance(pred_right, And):
        return And(get_pred_used(pred_left, pred_right.lh), get_pred_used(pred_left, pred_right.rh))
    elif isinstance(pred_right, Or):
        return Or(get_pred_used(pred_left, pred_right.lh),get_pred_used(pred_left, pred_right.rh))
    elif isinstance(pred_right, UnaryOp):
        pred_right.v = get_pred_used(pred_left, pred_right.v)
        return pred_right.v
    elif isinstance(pred_right, BinOp):
        if isinstance(pred_right.rh, Variable):
            cond_from_left = pred_left
            cond_new = copy.copy(get_from_left(cond_from_left,pred_right.rh.values))
            if cond_new is None:
                return True
            else:
                cond_new.lh  = pred_right.lh
            return cond_new
        else:
            return pred_right
    else:
        return pred_right


def get_cond(cond):
    if isinstance(cond.dependent_ops[0], SubpipeInput):
        return Expr(cond.expr)
    else:
        None


def generate_input_filters_general(op, infer, output_filter, output_schemas = {}, last_return = None):
    if isinstance(op, InitTable):
        return 'keep', [output_filter]
    if isinstance(op, SubpipeInput):
        return 'keep', [output_filter]
    if isinstance(op, AllAggregate):
        return 'rewrite', [True]
    if isinstance(op, ScalarComputation):
        if len(op.dependent_ops) == 1:
            cols = get_columns_used(output_filter)
            new_col = get_column_used_from_lambda(op.expr)[0]
            if new_col in cols:
                return 'rewrite', [get_filter_replacing_field(output_filter, {new_col: Expr(op.expr)})]
            else:
                return 'rewrite', [True]
        else:
            return 'rewrite', [True] * len(op.dependent_ops)
    if isinstance(op, CrosstableUDF):
        # get the left table and the right table
        sub_pipe = op.subpipeline
        inference_map = [{op:infer.inferences[j].sub_pipeline[i] for i,op in enumerate(path.operators)} for j,path in enumerate(op.subpipeline.paths)]
        if len(sub_pipe.paths) == 1:
            # row - left, table - right
            # it is possible that the right path is []
            left_path, right_path = rewrite_path(sub_pipe.paths[0])
            pred_on_new_col = get_pred_on_new_col(output_filter, op.new_col)
            pred_on_others = get_filter_removing_unused(output_filter, [op.new_col])
            if len(left_path) != 0:
                intput_filter_left_on_new_col = pushdown_subpipeline(left_path, inference_map[0], pred_on_new_col, output_schemas)[0]
                subpipeinput_left_from_cond = And(pred_on_others, intput_filter_left_on_new_col)
            else:
                subpipeinput_left_from_cond = False
            if len(right_path) != 0:
                subpipeinput_right_from_cond = pushdown_subpipeline(right_path, inference_map[0], pred_on_new_col, output_schemas)[0]
            else:
                subpipeinput_right_from_cond = False
            subpipeinput_right_from_cond = get_pred_used(subpipeinput_left_from_cond, subpipeinput_right_from_cond)
            infer.inferences[0].input_filters = [subpipeinput_left_from_cond, subpipeinput_right_from_cond]
            return 'rewrite', [subpipeinput_left_from_cond, subpipeinput_right_from_cond]
        else:
            subpipeinput_left = []
            subpipeinput_right = []
            for path_id,p in enumerate(sub_pipe.paths):
                left_path, right_path = rewrite_path(p)
                # first take a look at the condition is from which table
                path_cond = p.cond
                # seperate the predicates on op.new
                pred_on_new_col = get_pred_on_new_col(output_filter, op.new_col)
                pred_on_others = get_filter_removing_unused(output_filter, [op.new_col])
                # get the predicate on the new_col
                cond_input = path_cond
                # get the input of the cond
                while True:
                    cond_input = cond_input.dependent_ops[0]
                    if isinstance(cond_input, SubpipeInput):
                        break
                # if path_cond.dependent_ops[0].input_type == 'row':
                if cond_input.input_type == 'row':
                    # pushdown the pred_on_others to the input
                    input_filter_left_on_others = pred_on_others
                    cond = get_cond(path_cond)
                    output_filter_new = pred_on_new_col
                    if len(left_path) != 0:
                        intput_filter_left_on_new_col = pushdown_subpipeline(left_path, inference_map[path_id], output_filter_new, output_schemas)[0]
                        if cond is not None:
                            subpipeinput_left_from_cond = And(And(input_filter_left_on_others, intput_filter_left_on_new_col), cond)
                        else:
                            subpipeinput_left_from_cond = And(input_filter_left_on_others, intput_filter_left_on_new_col)
                    else:
                        subpipeinput_left_from_cond = False
                    subpipeinput_left.append(subpipeinput_left_from_cond)
                    if len(right_path) != 0:
                        subpipeinput_right_from_cond = pushdown_subpipeline(right_path, inference_map[path_id], output_filter_new, output_schemas)[0]
                    else:
                        subpipeinput_right_from_cond = False
                    subpipeinput_right.append(subpipeinput_right_from_cond)
                    subpipeinput_right_from_cond = get_pred_used(subpipeinput_left_from_cond, subpipeinput_right_from_cond)
                    infer.inferences[path_id].input_filters = [subpipeinput_left_from_cond, subpipeinput_right_from_cond]
                    if last_return == None:
                        return 'cond'+ str(path_id), [subpipeinput_left_from_cond, subpipeinput_right_from_cond]
                    if last_return == ('cond' + str(path_id -1)) and path_id < len(sub_pipe.paths):
                        return 'cond'+ str(path_id), [subpipeinput_left_from_cond, subpipeinput_right_from_cond]
                if cond_input.input_type == 'table':
                    input_filter_left_on_others = pred_on_others
                    if isinstance(path_cond, ScalarComputation):
                        cond = get_cond(path_cond)
                    output_filter_new = pred_on_new_col
                    if len(left_path) != 0:
                        intput_filter_left_on_new_col = pushdown_subpipeline(left_path, inference_map[path_id], output_filter_new, output_schemas)[0]
                        subpipeinput_left_from_cond = And(input_filter_left_on_others, intput_filter_left_on_new_col)
                    else:
                        subpipeinput_left_from_cond = False
                    subpipeinput_left.append(subpipeinput_left_from_cond)
                    if len(right_path) != 0:
                        subpipeinput_right_from_cond = pushdown_subpipeline(right_path, inference_map[path_id], output_filter_new, output_schemas)[0]
                        if cond is not None:
                            subpipeinput_right_from_cond = And(subpipeinput_right_from_cond, cond)
                    else:
                        subpipeinput_right_from_cond = False
                    subpipeinput_right.append(subpipeinput_right_from_cond)
                    subpipeinput_right_from_cond = get_pred_used(subpipeinput_left_from_cond, subpipeinput_right_from_cond)
                    infer.inferences[path_id].input_filters = [subpipeinput_left_from_cond, subpipeinput_right_from_cond]
                    if last_return == None:
                        return 'cond'+ str(path_id), [subpipeinput_left_from_cond, subpipeinput_right_from_cond]
                    if last_return == ('cond' + str(path_id -1)) and path_id < len(sub_pipe.paths):
                        return 'cond'+ str(path_id), [subpipeinput_left_from_cond, subpipeinput_right_from_cond]
                
                    infer.inferences[path_id].input_filters = [subpipeinput_left[-1], subpipeinput_right[-1]]
                # for each condition 
            # add candidate only takes one path
            subpipeinput_left = AllOr(*subpipeinput_left)
            subpipeinput_right = AllOr(*subpipeinput_right)
            subpipeinput_right = get_pred_used(subpipeinput_left, subpipeinput_right)
            return 'rewrite', [subpipeinput_left, subpipeinput_right]
    elif isinstance(op, CogroupedMap) or isinstance(op, GroupedMap):
        # group
        inference_map = [{op:infer.inferences[j].sub_pipeline[i] for i,op in enumerate(path.operators)} for j,path in enumerate(op.subpipeline.paths)]
        sub_pipe = op.subpipeline
        # split predicate as two parts: one on left and one on right
        group_key = set(op.subpipeline.paths[0].operators[0].group_key)
        pred_other = get_filter_removing_unused(output_filter, group_key)
        columns_used = set(get_columns_used(output_filter))
        pred_group = get_filter_removing_unused(output_filter, columns_used - group_key)
        temp_tup = generate_symbolic_table('temp-t', output_schemas[op], 1)[0]
        assert(check_predicate_equal(output_filter,And(pred_group, pred_other), temp_tup))
        infer.inferences[0].output_filter = pred_other
        
        subpipeinput_input_filter = pushdown_subpipeline(sub_pipe.paths[0].operators, inference_map[0], pred_other, output_schemas)
        # TODO: change the first value, this is to make it not break/    
        infer.input_filters = [And(pred_group, subpipeinput_input_filter[0])]    
        return 'rewrite', infer.input_filters        
    # print("=============================================================")
    if len(op.dependent_ops) == 1:
        p = get_updated_cols(op, output_filter, op.input_schema, output_schemas)
        # change type and group by is not handled.
        # if P is empty
        if len(p) == 0:
            if hasattr(op, 'condition'):
                if check_operator_in_cond(op.condition):
                    new_cond = get_cond_with_variable(op.condition)
                    return 'additional', [new_cond]             
                return 'additional', [And(op.condition, output_filter)]
            elif last_return == None:
            # keep it
                # print("keep")
                return 'keep', [output_filter]
        # if last return is rewrite or no rewrite rules applys.
        if last_return == 'rewrite' or check_rewrite_rules(op):
            # lastly, remove p
            return 'remove', [get_filter_removing_unused(output_filter, list(p))]
        # first try to rewrite it (rename, setitem, get_dummies) - three type
        # 1. first find if there is a name map
        if hasattr(op, 'name_map'):
            reversed_rename = dict([v,k] for k, v in op.name_map.items())
            return 'rewrite', [get_filter_replacing_field(output_filter, reversed_rename)]
        elif hasattr(op, 'new_column_names'):
            # this is for group by
            reversed_rename = dict([v,k] for k, v in op.new_column_names.items())
            return 'rewrite', [get_filter_replacing_field(output_filter, reversed_rename)]
        # handle change_type
        elif hasattr(op, 'target_type'):
            if op.target_type == 'str':
                str_lambda = 'lambda x: str(x["{}"])'.format(op.orig_col)
            elif op.target_type == 'int':
                str_lambda = 'lambda x: int(x["{}"])'.format(op.orig_col)
            elif op.target_type == 'float':
                str_lambda = 'lambda x: float(x["{}"])'.format(op.orig_col)
            elif op.target_type == 'datetime':
                str_lambda = 'lambda x: pd.to_datetime(x["{}"])'.format(op.orig_col)
            reversed_rename = {op.new_col: op.orig_col}
            output_filter = get_filter_replacing_field(output_filter, {op.orig_col: Expr(str_lambda, op.target_type)})
            return 'rewrite', [get_filter_replacing_field(output_filter, reversed_rename)]       
        # 2. find if there is a Expr
        elif hasattr(op, 'apply_func'):
            return 'rewrite', [get_filter_replacing_field(output_filter, {op.new_col: Expr(op.apply_func, op.return_type)})]
        elif hasattr(op, 'fill_value'):
            return 'rewrite', [get_filter_replacing_field(output_filter, {op.col: Expr(infer.f, get_variable_type(op.fill_value))}, [op.col, op.fill_value])]
        # handle split
        elif hasattr(op, 'new_col_names'):
            cols_used = get_columns_used(output_filter)
            for c in cols_used:
                if c in op.new_col_names:
                    pos = op.new_col_names.index(c)
                    #print(pos)
                    if op.regex is None and op.by is not None:
                        split_lambda = 'lambda x:x["{}"].split("{}")[{}]'.format(op.column_to_split, op.by, pos)
                    elif op.regex is not None:
                        split_lambda = 'lambda x:list(filter(None, re.split(r"{}", x["{}"])))[{}]'.format(op.regex, op.column_to_split, pos)
                    output_filter = get_filter_replacing_field(output_filter, {c: Expr(split_lambda, 'str')})
            return 'rewrite', [output_filter]
        # 3. find if there is a update of the output_schema, get_dummies, pivot, unpivot
        elif hasattr(op, 'value_name'):
            #  for unpivot
            columns_used = get_columns_used(output_filter)
            columns_in_idvar = set(columns_used).intersection(op.id_vars)
            return 'rewrite', [get_filter_replacing_for_unpivot(output_filter, columns_in_idvar, op.value_name, op.var_name, op.value_vars)]

        elif hasattr(op, 'output_schema'):
            # first compute the name_map:
            if hasattr(op, 'index_col'):
                columns_used = get_columns_used(output_filter)
                columns_not_in_group = set(columns_used).difference(set(op.index_col))
                return 'rewrite', [get_filter_replacing_unused_for_pivot(output_filter, columns_not_in_group, op.header_col, op.value_col)]
            else:
                return 'rewrite', [get_filter_replacing_with_schema(output_filter, op.cols, op.output_schema)]
        else:
            print("NOT HANDLED")
    elif len(op.dependent_ops) == 2 and hasattr(op, 'input_schema_left'):
        # handle join
        columns_left = set(op.input_schema_left.keys())
        columns_right = set(op.input_schema_right.keys())
        columns_left_new =[]
        columns_right_new =[] 
        for i in columns_left:
            if i in columns_right and i not in infer.merge_cols_left:
                columns_left_new.append(i+"_x")
            else:
                columns_left_new.append(i)
        for i in columns_right:
            if i in columns_left and i not in infer.merge_cols_right:
                columns_right_new.append(i+"_y")
            else:
                columns_right_new.append(i)
        columns_left = set(columns_left_new)
        columns_right = set(columns_right_new)
        columns_used_in_left = set(get_columns_used(output_filter)).intersection(columns_left).difference(set(infer.merge_cols_left))
        columns_used_in_right = set(get_columns_used(output_filter)).intersection(columns_right).difference(set(infer.merge_cols_right)) 
        if len(columns_used_in_right) == 0:
            input_filter_left = output_filter
        else:
            input_filter_left = get_filter_removing_unused(output_filter, columns_used_in_right)
        if len(columns_used_in_left) == 0:
            input_filter_right = output_filter
        else:
            input_filter_right = get_filter_removing_unused(output_filter, columns_used_in_left)
        replace_name_left = {}
        replace_name_right = {}
        for i in columns_used_in_left:
            if "_x" in i:
                new_col = i
                original_col = i[:-2]
                replace_name_left[new_col] = original_col
        for i in columns_used_in_right:
            if "_y" in i:
                new_col = i
                original_col = i[:-2]
                replace_name_right[new_col] = original_col
        if len(replace_name_left) >0:
            input_filter_left = get_filter_replacing_field(input_filter_left, replace_name_left)
        if len(replace_name_right) > 0:
            input_filter_right = get_filter_replacing_field(input_filter_right, replace_name_right)
        new_col_used_left = get_columns_used(input_filter_left)
        if not set(new_col_used_left).isdisjoint(set(op.merge_cols_right)):
            # naive solution
            input_filter_left = get_filter_replacing_field(input_filter_left, {op.merge_cols_right[0]:op.merge_cols_left[0]})
        new_col_used_right = get_columns_used(input_filter_right)
        if not set(new_col_used_right).isdisjoint(set(op.merge_cols_left)):
            # naive solution
            input_filter_right = get_filter_replacing_field(input_filter_right, {op.merge_cols_left[0]:op.merge_cols_right[0]})
        return 'join', [input_filter_left, input_filter_right]
    else:
        columns_notused_in_tables = []
        for i in range(len(op.dependent_ops)):
            if hasattr(op, 'input_schemas') or i==0:
                columns_i = [k for k,v in (op.input_schemas[i] if hasattr(op, 'input_schemas') else op.input_schema).items()] 
                columns_used_in_i = set(get_columns_used(output_filter)).intersection(columns_i)
                columns_not_used_i = set(get_columns_used(output_filter)) - columns_used_in_i
                columns_notused_in_tables.append(columns_not_used_i)
            else:
                columns_notused_in_tables.append(None)
        
        input_filters = []
        for i in range(len(op.dependent_ops)):
            if columns_notused_in_tables[i] is None:
                input_filters.append(True)
            elif len(columns_notused_in_tables[i]) == 0:
                input_filters.append(output_filter)
            else:
                input_filters.append(get_filter_removing_unused(output_filter, columns_notused_in_tables[i]))
        return 'multi', input_filters
        

def pushdown_subpipeline(pipelinepath, inference_map, output_filter, output_schemas):
    sub_ops = pipelinepath
    print(".................start subpipeline pushdown................")
    for op_id,op_j in reversed([(k1,v1) for k1,v1 in enumerate(sub_ops)]):
        print(op_j)
        if(op_j == sub_ops[-1]):
            output_filter_j = {op_j.dependent_ops[0]:output_filter}
        else:
            output_filter_j = generate_output_filter_from_previous(op_j, sub_ops)
        output_filter_new = AllOr(*list(output_filter_j.values()))
        #inference_j = op_j.get_inference_instance(output_filter_new)
        inference_j = inference_map[op_j]
        print("inference before = {}, after = {}".format(op_j.inference, inference_j))
        op_j.inference = inference_j
        inference_j.output_filter = output_filter_new
        last_return,inference_j.input_filters = generate_input_filters_general(op_j, inference_j, output_filter_new, output_schemas)
        print("--------input filter for {} = {}".format(type(op_j), inference_j.input_filters[0]))
    for op_id,op_j in reversed([(k1,v1) for k1,v1 in enumerate(sub_ops)]):
        # if isinstance(op_j, SubpipeInput) and op_j == dependent_op:
        if isinstance(op_j, SubpipeInput):
            print("The return value is")
            print(inference_map[op_j].input_filters[0])
            return inference_map[op_j].input_filters
            #return op_j.inference.input_filters


def check_predicate_equality(expr, op):
    # TODO: add one more rule....!!!
    if isinstance(expr, BoolOp):
        return check_predicate_equality(expr.lh, op) + check_predicate_equality(expr.rh, op)
    elif isinstance(expr, BinOp):
        if isinstance(expr.lh, Field) and expr.lh.col not in op.sort_order:
            return [expr.lh.col]
        elif isinstance(expr.lh, Field) and expr.lh.col in op.sort_order:
            if (expr.op == '=='):
                return [expr.lh.col]
            else:
                return []
        elif isinstance(expr.lh, Expr):
            # not sure, needs a verifier
            column_used  = get_column_used_from_lambda(expr.lh.expr)
            return [column_used]
    elif isinstance(expr, UnaryOp):
        return check_predicate_equality(expr.v, op)

def print_input_filters(infer):
    for i in infer.input_filters:
        print(i)


def output_filter_rewrite(output_schemas, output_filter, op, trials):
    if len(op.dependent_ops) == 1:
        p = get_updated_cols(op, output_filter, op.input_schema, output_schemas)
        # for join condition
        candidates = generate_candidate(output_filter, op, p)
        print(candidates)
        # for debugging setting, add an nonempty constriant to the filter.
        if len(candidates) == trials:
            return -1, []
        op.constraints.append(FilterNonEmpty(output_filter))
        candidate = candidates[trials]
        print("validating candidate")
        print(candidate)
        trials += 1
        result = check_output_filter_rewrite(output_schemas[op], output_filter, candidate, op.constraints)
        if result:
            print("get a new output filter {} after {} attempts".format(candidate, trials))
            return trials, [candidate]
        else:
            return trials, []
    else:
        return -1, []


def get_ops(op):
    res = []
    res.append(op)
    cur_op = copy.copy(op)
    while True:
        next = cur_op.dependent_ops
        if len(next) ==1:
            res = res + next
            cur_op = res[-1]          
        else: 
            break
    res.reverse()
    return res

def get_operator_output_variable_name(i):
    return 'df{}'.format(i)

def generate_code_snapshot(ops, output_filter, dump_path):
    lines = []
    input_variable_names = {}
    for i,op in enumerate(ops):
        output_var = get_operator_output_variable_name(i)
        input_variable_names[op] = output_var
        lines.append(op.to_python(output_var, input_variable_names))
        if i == len(ops)-1:
            lines.append("final_output = {}[{}.apply(lambda row: {}, axis=1)]".format(output_var, output_var, pred_to_python_using_lambda(output_filter, output_var)))
            lines.append("pickle.dump(final_output, open('{}/result_snapshot.p', 'wb'))".format(dump_path))
    return '\n'.join(lines)    


def get_snapshot_df(op, output_filter):
    # get the snapshot df, apply filter on the snapshot
    ops = get_ops(op)
    code = generate_code_snapshot(ops, output_filter, 'temp/')
    #print(code)
    exec(code)
    output = pickle.load(open('temp/'+"/result_snapshot.p", 'rb'))
    return output   


def get_output_filter_variable_sql(output_schema_op, op, output_filter, query_id):
    pred = []
    columns_used = get_columns_used(output_filter)
    output_cols = set(output_schema_op.keys())
    p = set(columns_used).intersection(output_cols)
    # for groupby
    output_df = get_snapshot_df_sql(query_id)
    for k, v in output_schema_op.items():
        if (v in p):
            value = output_df[k][0]
            pred.append(BinOp(Field(k), '==', Constant(value, typ=v)))
        else:
            if (k == 'index') or k not in output_df.columns:
                continue
            values = list(output_df[k].unique())
            if v == 'int' or v == 'float':
                var = z3.Int('some-v')
            elif v == 'str':
                var = z3.String('some-v')
            pred.append(BinOp(Field(k), '==', Variable(var, values = values, typ=v)))
    return AllAnd(*pred)

def get_cond_str(expr):
    if isinstance(expr, And):
        return get_cond_str(expr.lh) + " and " + get_cond_str(expr.rh)
    if isinstance(expr, BinOp):
        if isinstance(expr.rh, Constant):
            if expr.rh.typ == 'int':
                return str(expr.lh)[1:-1] + '=' + str(expr.rh.v)
            if expr.rh.typ == 'str':
                return str(expr.lh)[1:-1] + '=' + "'" + str(expr.rh.v) + "'"
        elif isinstance(expr.rh, Variable):
            return_str = ""
            for i in expr.rh.values:
                if expr.rh.typ == 'str':
                    if i != expr.rh.values[-1]:
                        return_str = return_str + str(expr.lh)[1:-2] + '=' + "'" + str(i) + "'" + " or "
                    else:
                        return_str = return_str + str(expr.lh)[1:-2] + '=' + "'" + str(i) + "'" 
                else:
                    if i != expr.rh.values[-1]:
                        return_str = return_str + str(expr.lh)[1:-2] + '=' + str(i) + " or "
                    else:
                        return_str = return_str + str(expr.lh)[1:-2] + '=' + str(i)
            return return_str



def get_snapshot_df_sql(query_id):
    conn = psycopg2.connect(dbname = 'testdb', user = 'test', password = '0000', host ='127.0.0.1', port = '5432')
    name = '/datadrive/yin/predicate_pushdown_for_lineage_tracking/test/TPCH/TPC-H V3.0.1/dbgen/queries/' + query_id
    cursor = conn.cursor()
    cursor.execute(open(name, "r").read())
    data = cursor.fetchall()
    col_names = [desc[0] for desc in cursor.description]
    result = pd.DataFrame(data, columns= col_names)
    conn.commit()
    conn.close()
    return result

def get_output_filter_variable(output_schema_op, op, input_cols, output_filter):
    pred = []
    columns_used = get_columns_used(output_filter)
    output_cols = set(output_schema_op.keys())
    # TODO: handle split
    column_new = output_cols - input_cols
    p = set(columns_used).intersection(column_new)
    # for groupby
    if hasattr(op, 'new_column_names'):
        agg_col = set(op.new_column_names.values())
        p = p.union(agg_col)
        print(agg_col)
    # pred_full = []
    output_df = get_snapshot_df(op, output_filter)
    print(output_df.columns)
    for k, v in output_schema_op.items():
        if (v in p):
            value = output_df[k][0]
            pred.append(BinOp(Field(k), '==', Constant(value, typ=v)))
        else:
            if (k == 'index') or k not in output_df.columns:
                continue
            values = list(output_df[k].unique())
            if v == 'int' or v == 'float':
                var = z3.Int('some-v')
            elif v == 'str':
                var = z3.String('some-v')
            pred.append(BinOp(Field(k), '==', Variable(var, values = values, typ=v)))
    return AllAnd(*pred)




def snapshot_generate_predicate(output_schemas, op, output_filter):
    if isinstance(op, InnerJoin) or isinstance(op, LeftOuterJoin):
        table_left = output_schemas[op.dependent_ops[0]]
        input_filter_left = get_output_filter_variable(table_left, op, set(op.input_schema_left.keys()), output_filter)
        table_right = output_schemas[op.dependent_ops[1]]
        input_filter_right = get_output_filter_variable(table_right, op, set(op.input_schema_right.keys()), output_filter)
        return [input_filter_left, input_filter_right]
    else:
        input_schema_op = output_schemas[op.dependent_ops[0]]
        input_filter = get_output_filter_variable(input_schema_op, op, set(op.input_schema.keys()), output_filter)
        return [input_filter]


def snapshot_generate_predicate_sql(output_schemas, op, output_filter, query_id):
    if isinstance(op, InnerJoin) or isinstance(op, LeftOuterJoin):
        table_left = output_schemas[op.dependent_ops[0]]
        input_filter_left = get_output_filter_variable_sql(table_left, op,output_filter, query_id)
        table_right = output_schemas[op.dependent_ops[1]]
        input_filter_right = get_output_filter_variable_sql(table_right, op, output_filter, query_id)
        return [input_filter_left, input_filter_right]
    else:
        input_schema_op = output_schemas[op.dependent_ops[0]]
        input_filter = get_output_filter_variable_sql(input_schema_op, op, output_filter, query_id)
        return [input_filter]



# heuristic to generate candidate filters
def generate_candidate(output_filter, op, problematic_cols):
    res = []
    # removing all problematic cols (unique constraints)
    res.append(get_filter_removing_unused(output_filter, problematic_cols))
    # using the FD contrains
    output_filter_c = output_filter
    for c in op.constraints:
        if isinstance(c, FunctionalDependency):
            if c.dependent_column in problematic_cols:
                # not sure if it is correct
                output_filter_c = get_filter_replacing_field(output_filter_c, {c.dependent_column: Expr(c.expr)})
                problematic_cols.remove(c.dependent_column)
    if len(problematic_cols) == 0:
        res.append(output_filter_c)
    return res



def change_type(expr, type, orig_col):
    if type == 'str':
        if isinstance(expr, BinOp):
            if isinstance(expr.rh, Constant) and expr.lh.col == orig_col:
                expr.rh.typ = 'str'
                str_lambda = 'lambda xxx__:str(xxx__[\'{}\'])'.format(expr.lh.col)
                expr.lh = Expr(str_lambda)
            return expr
        elif isinstance(expr, UnaryOp):
            return change_type(expr.v, 'str', orig_col)
        elif isinstance(expr, BoolOp):
            expr.lh = change_type(expr.lh, 'str', orig_col)
            expr.rh = change_type(expr.rh, 'str', orig_col)
            return expr
        else:
            return expr       
    elif type == 'int':
        if isinstance(expr, BinOp):
            if isinstance(expr.rh, Constant) and expr.lh.col == orig_col:
                expr.rh.typ = 'int'
                str_lambda = 'lambda xxx__:int(xxx__[\'{}\'])'.format(expr.lh.col)
                expr.lh = Expr(str_lambda)
            return expr
        elif isinstance(expr, UnaryOp):
            return change_type(expr.v,'int', orig_col)
        elif isinstance(expr, BoolOp):
            expr.lh = change_type(expr.lh,'int', orig_col)
            expr.rh = change_type(expr.rh,'int', orig_col)
            return expr
        else:
            return expr 
    elif type == 'datetime':
        if isinstance(expr, BinOp):
            if isinstance(expr.rh, Constant) and expr.lh.col == orig_col:
                expr.rh.typ = 'datetime'
                str_lambda = 'lambda xxx__:pd.to_datetime(xxx__[\'{}\'])'.format(expr.lh.col)
                expr.lh = Expr(str_lambda)
            return expr
        elif isinstance(expr, UnaryOp):
            return change_type(expr.v, 'datetime', orig_col)
        elif isinstance(expr, BoolOp):
            expr.lh = change_type(expr.lh, 'datetime', orig_col)
            expr.rh = change_type(expr.rh, 'datetime', orig_col)
            return expr
        else:
            return expr 
    else:
        return expr 



