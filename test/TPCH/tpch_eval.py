import sys
sys.path.append("/datadrive/cong/predicate_pushdown_for_lineage_tracking/")

import z3
import dis
from interface import *
from util import *
import random
# from constraint import *
from predicate import *
from generate_input_filters import generate_input_filters_general, generate_output_filter_from_previous
import os
from table_constraint import *
from input_filter_baseline import get_input_filter_baseline
import sys
import numpy as np

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


output_filter = True
output_schemas = generate_output_schemas(ops)
superset = []
use_baseline = False
assertion=False
scale_data_only = False
if len(sys.argv) > 2 and 'base' in sys.argv[2]:
    use_baseline = True
    if len(sys.argv) > 3:
        assertion=True
if len(sys.argv) > 2 and 'scale' in sys.argv[2]:
    scale_data_only = True
    
    
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
    print("output filter for ({}) {} is {}".format(op_id, type(inference), inference.output_filter))
    last_return = None
    if use_baseline:
        print(type(inference))
        inference.input_filters = get_input_filter_baseline(output_filter, op_i, inference, assertion)
        print(inference.input_filters[0])
        if all([type(p) is bool and p==True for p in inference.input_filters]):
            superset.append(op_i)
    else:
        candidates = generate_input_filters_general(op_i, inference, output_filter)
        found = False
        for candidate in candidates:
            print("\tcandidate input = {}".format(' / '.join([str(x) for x in candidate])))
            inference.input_filters = candidate
            if inference.check_small_model() and inference.verify_correct():
                found = True
                break
            if inference.check_small_model(check_superset=True) and inference.verify_correct(check_superset=True):
                superset.append(op_i)
                found = True
                break
        if found == False:
            inference.input_filters = [True for i in range(len(inference.input_filters))]

elapse = time.time()-start
print("pushed to after {}".format(last_op))
print("Finish within {} sec".format(elapse))
