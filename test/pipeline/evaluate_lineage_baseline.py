import sys
sys.path.append("/datadrive/yin/predicate_pushdown_for_lineage_tracking/")

import z3
import dis
from interface import *
from util import *
import random
from predicate import *
from eval_util import *
from lineage_tracking_baseline import *
from compare_pushdown_result import get_operator_output_variable_name
import numpy as np
import re
import functools
import time

start_2=time.time()

NB_path = sys.argv[1]
pipe_code, ops = read_pipeline_code(NB_path)
pipe_code.append("ops = [{}]".format(','.join(ops)))
exec(''.join(pipe_code))
output_schemas = generate_output_schemas(ops)

def generate_script_with_lineage_tracking(ops, dump_path):
    lines = []
    input_variable_names = {}
    table_id = 0
    for i,op in enumerate(ops):
        if isinstance(op, InitTable):
            table_id += 1
        output_var = get_operator_output_variable_name(i)
        input_variable_names[op] = output_var
        lines.append(pandas_code_with_lineage_tracking_oneop(op, output_var, input_variable_names, table_id))
        if i == len(ops)-1: 
            lines.append("print({}.head())\n".format(output_var))
            lines.append("print({})\n".format(output_var))
        #    lines.append("pickle.dump({}, open('{}/result_lineage.p', 'wb'))".format(output_var, dump_path))
    return groupby_sum_code + "\n" + '\n'.join(lines) 

code = generate_script_with_lineage_tracking(ops, NB_path)
print(code)
exec(code)


end_2 = time.time()
print("The time for baseline is")
print(end_2-start_2)