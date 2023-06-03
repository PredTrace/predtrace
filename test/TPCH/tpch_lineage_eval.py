import sys
sys.path.append("/datadrive/yin/predicate_pushdown_for_lineage_tracking/")

import z3
import dis
from interface import *
from util import *
import random
# from constraint import *
from predicate import *
from generate_input_filters import generate_input_filters_general, generate_output_filter_from_previous, snapshot_generate_predicate_sql, get_cond_str, output_filter_rewrite
import os
from table_constraint import *
from input_filter_baseline import get_input_filter_baseline
import sys
import numpy as np
from compare_pushdown_result import get_output_filter, get_output_filter_only_from_schema, get_output_schema_tpch
import psycopg2

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

NB_path = sys.argv[1]
pipe_code, orig_ops = read_pipeline_code(NB_path)
ops = []
pipe_code.append("\nops = [{}]".format(','.join(orig_ops)))
print(''.join(pipe_code))
exec(''.join(pipe_code))

query_id1 = 'd8.sql'
query_id2 = 'd8_1.sql'

output_schemas = generate_output_schemas(ops)


output_filter = get_output_schema_tpch(query_id1, output_schemas[ops[-1]])

print(output_filter)

superset = []
is_superset = []
use_baseline = False
assertion=False
scale_data_only = False
debug = False
for op in ops:
    get_constraint(op)
if len(sys.argv) > 2:
    debug=True
    
import time
start = time.time()
last_op = None



for op_id,op_i in reversed([(k1,v1) for k1,v1 in enumerate(ops)]):
    if(op_i == ops[-1]):
        output_filter_i = {None:output_filter}
    else:
        output_filter_i = generate_output_filter_from_previous(op_i, ops)
    output_filter = AllOr(*list(output_filter_i.values()))
    inference = op_i.get_inference_instance(output_filter)
    last_return = None
    rewrite = False
    snapshot = False
    if use_baseline:
        print(type(inference))
        inference.input_filters = get_input_filter_baseline(output_filter, op_i, inference, assertion)
        print(inference.input_filters[0])
        if all([type(p) is bool and p==True for p in inference.input_filters]):
            superset.append(op_i)
    else:
        last_filters = []
        while True:
            last_return, inference.input_filters = generate_input_filters_general(op_i, inference, output_filter, output_schemas, last_return)
            if debug:
                print("output schema = {}".format(output_schemas[op_i]))
                print("output filter of {} = {}".format(type(op_i), inference.output_filter))
                print("To verify {}".format(inference.input_filters[0]))
            if any([all([str(history[i])==str(inference.input_filters[i]) for i in range(len(inference.input_filters))]) for history in last_filters]):
                inference.input_filters = [True for i in range(len(inference.input_filters))]
                print("CANNOT PUSHDOWN， TRY FILTER REWRITE OR SNAPSHOT.")
                rewrite = True
                snapshot = True
                break
            last_filters.append(inference.input_filters)
            if inference.check_small_model() and inference.verify_correct():
                if debug:
                    print("The correct input filter is")
                    print(inference.input_filters[0])
                if determine_superset_simple(op_i, inference.output_filter):
                    print("LINEAGE SUPERSET")
                    snapshot = True
                break
            if inference.check_small_model(check_superset=True) and inference.verify_correct(check_superset=True):
                if debug:
                    print("Break because of superset")
                    print(inference.input_filters[0])
                is_superset.append(type(op_i))
                if determine_superset_simple(op_i, inference.output_filter):
                    print("LINEAGE SUPERSET")
                    snapshot = True
                break
        if rewrite:
            trials = 0
            print("Try output filter rewrite")
            while True:
                trials, output_filter_new = output_filter_rewrite(output_schemas, output_filter, op_i, trials)
                if len(output_filter_new) != 0:
                    inference.output_filter = output_filter_new
                    inference.input_filters = output_filter_new
                    snapshot = False
                    break
                if trials == -1:
                    break        
        if snapshot:
            print("take a snapshot.")
            inference.input_filters = snapshot_generate_predicate_sql(output_schemas, op_i, output_filter, query_id2)  

elapse = time.time()-start
print("pushed to after {}".format(last_op))
print("Finish within {} sec".format(elapse))

def get_input_records_sql(ops):
    for op in ops:
        if isinstance(op, InitTable):
            tbl = op.datafile_path.strip('.csv')
            tbl = tbl.strip('data/')
            print(tbl)
            print(op.inference.input_filters[0])
            cond_str = get_cond_str(op.inference.input_filters[0])
            sql_str = "select count(*) from {} where {};".format(tbl, cond_str)
            print(sql_str)
            
            conn = psycopg2.connect(dbname = 'testdb', user = 'test', password = '0000', host ='127.0.0.1', port = '5432')
            cursor = conn.cursor()
            cursor.execute(sql_str)
            data = cursor.fetchone()
            print(data)

