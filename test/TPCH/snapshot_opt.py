import sys
sys.path.append("../../")
from interface import *
from table_constraint import *

def get_and_pred(cols):
    return AllAnd(*[BinOp(Field(col), '==', Variable('?')) for col in cols])

def replace_scalar_with_var(pred):
    if isinstance(pred, BinOp):
        if isinstance(pred.rh, ScalarComputation):
            return BinOp(pred.lh, pred.op, Variable('?'))
        else:
            return pred
    elif isinstance(pred, And):
        return And(replace_scalar_with_var(pred.lh),replace_scalar_with_var(pred.rh))
    elif isinstance(pred, Or):
        return Or(replace_scalar_with_var(pred.lh),replace_scalar_with_var(pred.rh))
    else:
        return pred

def getsubpipe_cols_used_in_lefttable(op):
    cols_in_subpipe = set()
    subpipe_ops = op.subpipeline.paths[0].operators
    for op1 in subpipe_ops:
        if isinstance(op1, ScalarComputation):
            cols_in_lambda = get_column_used_from_lambda(op1.expr)
            if len(cols_in_lambda) > 0:
                cols_in_subpipe = cols_in_subpipe.union(set(cols_in_lambda))
    for temp_op in ops:
        if any([dep==op for dep in temp_op.dependent_ops]) and isinstance(temp_op, Filter):
            cols_involved = get_columns_used(temp_op.condition)
            if any([col_==op.new_col for col_ in cols_involved]):
                for col_ in cols_involved:
                    if col_!=op.new_col:
                        cols_in_subpipe.add(col_)
    return cols_in_subpipe

def columns_needed(op, left=True):
    if isinstance(op, InnerJoin):
        return set(op.merge_cols_left).union(set(op.merge_cols_right))
    elif isinstance(op, GroupBy):
        return set(op.groupby_cols + [k for k,v in op.aggr_func_map.items()])
    elif isinstance(op, SortValues):
        return set(op.cols)
    elif isinstance(op, DropColumns):
        return set(op.cols)
    elif isinstance(op, Filter):
        return set(get_columns_used(op.condition))
    elif isinstance(op, AllAggregate):
        return get_column_used_from_lambda(op.aggr_func)
    elif isinstance(op, ScalarComputation):
        return get_column_used_from_lambda(op.expr)
    elif isinstance(op, CrosstableUDF):
        if left:
            return getsubpipe_cols_used_in_lefttable(op)
        else:
            cols = set()
            for p in op.subpipeline.paths:
                for op1 in p.operators:
                    if not isinstance(op1, ScalarComputation):
                        cols = cols.union(columns_needed(op1))
            return cols
    elif isinstance(op, TopN):
        return op.sort_order
    elif isinstance(op, Rename):
        return [k for k,v in op.name_map.items()]
    elif isinstance(op, SetItem):
        return get_columns_used(op.apply_func)
    else:
        assert(False)


def filter_to_sql(cond):
    if isinstance(cond, And):
        return "{} and {}".format(filter_to_sql(cond.lh), filter_to_sql(cond.rh))
    if isinstance(cond, Or):
        return "({} or {})".format(filter_to_sql(cond.lh), filter_to_sql(cond.rh))
    elif isinstance(cond, BinOp):
        if isinstance(cond.rh, ScalarComputation):
            return get_column_used_from_lambda(cond.rh.expr)[0]
        else:
            return '{} {} {}'.format(filter_to_sql(cond.lh), cond.op, filter_to_sql(cond.rh))
    elif isinstance(cond, Field):
        return cond.col
    elif isinstance(cond, Constant):
        return cond.v
    else:
        print(type(cond))
        assert(False)
def generate_sql(op, snapshot_name=None, projections=[]):
    stack = [op]
    sort_keys = []
    group_keys = []
    filters = []
    tables = []
    limit = None
    processed = set()
    while len(stack) > 0:
        cur_op = stack.pop(0)
        processed.add(cur_op)
        print("\t{}".format(type(cur_op)))
        if isinstance(cur_op, InnerJoin):
            for i in range(len(cur_op.merge_cols_left)):
                filters.append('{} = {}'.format(cur_op.merge_cols_left[i],cur_op.merge_cols_right[i]))
        elif isinstance(cur_op, GroupBy):
            group_keys = cur_op.groupby_cols
        elif isinstance(cur_op, SortValues):
            sort_keys = cur_op.cols
        elif isinstance(cur_op, Filter):
            filters.append(filter_to_sql(cur_op.condition))
        elif isinstance(cur_op, InitTable):
            tables.append(cur_op.datafile_path.replace('data','').replace('/','').replace('.csv',''))
        elif isinstance(cur_op, DropColumns):
            projections = cur_op.cols
        elif isinstance(cur_op, TopN):
            limit = " LIMIT {}".format(cur_op.Nrows)
        elif isinstance(cur_op, SetItem):
            projections = projections + list(get_columns_used(cur_op.apply_func))
        elif isinstance(cur_op, AllAggregate):
            projections.append('xxx')
        elif isinstance(cur_op, ScalarComputation):
            continue

        elif isinstance(cur_op, CrosstableUDF):
            pass

        for dep in cur_op.dependent_ops:
            if dep not in processed:
                stack.append(dep)
    sql = "SELECT {}{} FROM {}".format(','.join(projections), '' if snapshot_name is None else ' INTO {}'.format(snapshot_name), ','.join(tables))
    if len(filters) > 0:
        sql += " WHERE {}".format(" and ".join([f for f in filters]))
    if len(group_keys) > 0:
        sql += " GROUP BY {}".format(','.join(group_keys))
    if len(sort_keys) > 0:
        sql += " ORDER BY {}".format(','.join(sort_keys))
    if limit:
        sql += limit
    return sql

snapshots = []

def get_pushdown_pred(F, constraint_map, op): 
    cols = get_columns_used(F)
    if isinstance(op, InnerJoin):
        if all([op.merge_cols_left[i] in cols or op.merge_cols_right[i] in cols for i in range(len(op.merge_cols_left))]):
            cols = set(cols)
            cols = cols.union(set(op.merge_cols_left))
            cols = cols.union(set(op.merge_cols_right))
            cols_left = list(filter(lambda col: col in cols, [col for col in op.input_schema_left]))
            cols_right = list(filter(lambda col: col in cols, [col for col in op.input_schema_right]))
            return [get_and_pred(cols_left), get_and_pred(cols_right)]
        else:
            return None
    elif isinstance(op, GroupBy):
        if all([g in cols for g in op.groupby_cols]):
            return get_and_pred(op.groupby_cols)
        elif all([c in op.groupby_cols for c in cols]):
            return F
        else:
            return None
    elif isinstance(op, SortValues):
        return F
    elif isinstance(op, DropColumns):
        return get_and_pred(list(set(op.cols).intersection(cols)))
    elif isinstance(op, Filter) :
        if all([c1 in cols for c1 in constraint_map[op].columns]):
            return F
        else:
            return And(F, replace_scalar_with_var(op.condition))
    elif isinstance(op, TopN) or isinstance(op, InitTable):
        return F
    elif isinstance(op, Rename):
        reversed_rename = dict([v,k] for k, v in op.name_map.items())
        return get_filter_replacing_field(F, reversed_rename)
    elif isinstance(op, SetItem):
        return F
    elif isinstance(op, AllAggregate):
        return F
    elif isinstance(op, ScalarComputation):
        return [F for i in range(len(op.dependent_ops))]
    elif isinstance(op, CrosstableUDF):
        # if any left table columns used is not in F, need a snapshot
        cols_in_subpipe = getsubpipe_cols_used_in_lefttable(op)
        if any([c1 not in cols for c1 in cols_in_subpipe]):
            return None

        subpipe_ops = op.subpipeline.paths[0].operators
        op_output_F = {subpipe_ops[-1]:True}
        snapshot_ops = []
        input_F = {}
        right_table_pred = None
        for op1 in reversed(subpipe_ops):
            if (isinstance(op1, ScalarComputation) and isinstance(op1.dependent_ops[0], SubpipeInput)) \
                or isinstance(op1, SubpipeInput):
                if isinstance(op1, SubpipeInput) and op1.input_type == 'table':
                    right_table_pred = op_output_F[op1]
                continue
            print("\tsubpip OP : {}".format(type(op1)))
            pred = get_pushdown_pred(op_output_F[op1], constraint_map, op1)
            if pred is None: # need snapshot
                print("\t\t^^^^ needs a snapshot ^^^^")
                c = constraint_map[op1]
                row_sel_pred1 = get_and_pred(c.columns)
                snapshots.append((op,c.columns))
                pred = get_pushdown_pred(row_sel_pred1, constraint_map, op1)
            pred = pred if type(pred) is list else [pred]
            print("\t\tafter pushdown: {}".format(' / '.join([str(p) for p in pred])))
            for j,dep in enumerate(op1.dependent_ops):
                op_output_F[dep] = pred[j] if dep not in op_output_F else Or(op_output_F[dep],pred[j])
            if len(op.dependent_ops) == 0:
                input_F[op1] = pred[0]
        return [F, right_table_pred]


def constraint_propagate(op, constraint_map):
    if isinstance(op, InnerJoin):
        cols = set()
        for dep in op.dependent_ops:
            cols = cols.union(constraint_map[dep].columns)
        cols = cols.union(op.merge_cols_left)
        cols = cols.union(op.merge_cols_right)
        return UniqueConstraint(list(cols))
    elif isinstance(op, GroupBy):
        return UniqueConstraint(list(op.groupby_cols))
    elif isinstance(op, SortValues):
        return constraint_map[op.dependent_ops[0]]
    elif isinstance(op, DropColumns):
        return constraint_map[op.dependent_ops[0]]
    elif isinstance(op, Filter) or isinstance(op, TopN):
        return constraint_map[op.dependent_ops[0]]
    elif isinstance(op, SubpipeInput):
        return constraint_map[op.dependent_ops[0]]
    elif isinstance(op, CrosstableUDF):
        row_op = None
        for p in op.subpipeline.paths:
            for op1 in p.operators:
                constraint_map[op1] = constraint_propagate(op1, constraint_map)
                if isinstance(op1, SubpipeInput) and op1.input_type == 'row':
                    row_op = op1
        return constraint_map[row_op]
    elif isinstance(op, SetItem) or isinstance(op, Rename):
        return constraint_map[op.dependent_ops[0]]
    else:
        return UniqueConstraint([])



def get_snapshot_projection(pipeline, constraint_map, snapshot_op):
    snapshot_op_id = 0
    for i,op in enumerate(pipeline):
        if op == snapshot_op:
            snapshot_op_id = i
    cols = set([])
    for op in enumerate(pipeline[snapshot_op_id+1:]):
        cols = cols.union(columns_needed(op))
    # union primary keys of the snapshot
    cols = cols.union(set(constraint_map[snapshot_op].columns))
    return cols
    

def get_new_queries(pipeline):
    queries = []
    print("\n====")
    print("Queries:")
    for i,op in enumerate(pipeline):
        if any([op == op1 for op1,columns in snapshots]):
            snapshot,columns = list(filter(lambda x:x[0]==op, snapshots))[0]
            sql = generate_sql(op, 'snapshot_{}'.format(i), columns)
            print(sql)
            queries.append(sql)
            for j in range(i+1, len(pipeline)):
                for k,dep in enumerate(pipeline[j].dependent_ops):
                    if dep == op:
                        pipeline[j].dependent_ops[k] = InitTable('snapshot_{}'.format(i),skip_read=True)
        if isinstance(op, CrosstableUDF):
            pass
    queries.append(generate_sql(pipeline[-1]))
    print(queries[-1])


def get_table_keys(dataf_name):
    f = dataf_name.split('/')[-1].replace('.csv','')
    if f == 'customer':
        return ['c_custkey']
    elif f == 'lineitem':
        return ['l_orderkey','l_linenumber']
    elif f == 'orders':
        return ['o_orderkey']
    elif f == 'nation':
        return ['n_nationkey']
    elif f == 'part':
        return ['p_partkey']
    elif f == 'region':
        return ['r_regionkey']
    elif f == 'supplier':
        return ['s_suppkey']
    elif f == 'partsupp':
        return ['ps_suppkey','ps_partkey']
    else:
        assert(False)
        

def get_output_schema(pipeline, op):
    for op1 in pipeline:
        if any([dep==op for dep in op1.dependent_ops]):
            if len(op1.dependent_ops) == 1:
                return op1.input_schema
            for i,dep in enumerate(op1.dependent_ops):
                if dep == op:
                    return op1.input_schema_left if i == 0 else op1.input_schema_right

def get_columns_used_later(pipeline, op):
    start = False
    ret = set()
    for i,op1 in enumerate(pipeline):
        if op == op1:
            start = True
        if start:
            ret = ret.union(columns_needed(op1))
    return ret


def snapshot_original(pipeline):
    constraint_map = {} # constraint of the output of op
    for op in pipeline:
        if isinstance(op, InitTable):
            constraint_map[op] = UniqueConstraint(get_table_keys(op.datafile_path))
    for op in pipeline:
        if op not in constraint_map:
            constraint_map[op] = constraint_propagate(op, constraint_map)
        print("constraint after {} is {}".format(type(op), constraint_map[op].columns))
    
    c = constraint_map[pipeline[-1]]
    if c is not None:
        row_sel_pred = get_and_pred(c.columns)
    else: 
        #row_sel_pred = ??
        assert(False)
    
    print("row-sel pred = {}".format(row_sel_pred))

    op_output_F = {pipeline[-1]:row_sel_pred}
    snapshot_ops = []
    input_F = {}
    for op in reversed(pipeline):
        print("OP : {}".format(type(op)))
        pred = get_pushdown_pred(op_output_F[op], constraint_map, op)
        if pred is None: # need snapshot
            print("\t^^^^ needs a snapshot ^^^^")
            c = constraint_map[op]
            op_output_schema = get_output_schema(pipeline, op)
            columns_used_later = get_columns_used_later(pipeline, op)
            sel_cols = set(c.columns + list(filter(lambda x: x in op_output_schema, columns_used_later)))
            row_sel_pred1 = get_and_pred(sel_cols)
            snapshots.append((op, sel_cols))
            pred = get_pushdown_pred(row_sel_pred1, constraint_map, op)
        pred = pred if type(pred) is list else [pred]
        print("\tafter pushdown: {}".format(' / '.join([str(p) for p in pred])))
        for j,dep in enumerate(op.dependent_ops):
            op_output_F[dep] = pred[j] if dep not in op_output_F else Or(op_output_F[dep],pred[j])
        if len(op.dependent_ops) == 0:
            input_F[op] = pred[0]
        
    
def read_pipeline_code(pipeline_path):
    pipe_code = []
    ops = []
    starts = False
    start_subop = False
    for line in open(pipeline_path):
        if len(line) > 1 and ' = ' in line and line.split(' = ')[1].lstrip()[0].isupper():
            starts = True
        if starts and len(line) > 1:
            if line.startswith('ops = ['):
                break
            if line.startswith('#') or line.startswith('"""'):
                continue
            if ' = ' not in line:
                if not line.replace('\n','').rstrip().endswith(')'):
                    pipe_code.append(line.replace('\n', ' ').replace('\\',' '))
                else:
                    pipe_code.append(line)
            else:
                pipe_code.append(line)
                op_name = line.split(' = ')[0]
                if op_name.startswith('sub'):
                    start_subop = True
                if line.split(' = ')[1].startswith("CrosstableUDF") or line.split(' = ')[1].startswith("CogroupedMap") or line.split(' = ')[1].startswith("GroupedMap"):
                    start_subop = False
                if start_subop == False and line.split(' = ')[1].lstrip()[0].isupper():
                    ops.append(op_name)
        
    return pipe_code, ops



query_path = sys.argv[1]
pipe_code, orig_ops = read_pipeline_code(query_path)
ops = []
pipe_code.append("\nops = [{}]".format(','.join(orig_ops)))
exec(''.join(pipe_code))
output_schemas = generate_output_schemas(ops)
pipeline = ops
snapshot_original(pipeline)
get_new_queries(pipeline)

