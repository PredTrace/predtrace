import os
import sys
def read_pipeline_code(NB_path):
    pipe_code = []
    ops = []
    pipeline_path = os.path.join(NB_path, 'pipeline.py')
    starts = False
    start_subop = False
    for line in open(pipeline_path):
        if len(line) > 1 and ' = ' in line and line.split(' = ')[1].lstrip()[0].isupper():
            starts = True
        if starts and len(line) > 1:
            if line.startswith('ops = ['):
                break
            if line.startswith('#'):
                continue
            if ' = ' in line and line.split(' = ')[1].startswith('InitTable('):
                pipe_code.append(line.replace('InitTable("','InitTable("{}/'.format(NB_path)).replace("InitTable('","InitTable('{}/".format(NB_path)))
                ops.append(line.split(' = ')[0])
            elif ' = ' not in line:
                if not line.replace('\n','').rstrip().endswith(')'):
                    pipe_code.append(line.replace('\n', ' ').replace('\\',' '))
                else:
                    pipe_code.append(line)
            else:
                pipe_code.append(line)
                op_name = line.split(' = ')[0]
                if op_name.startswith('sub_'):
                    start_subop = True
                if line.split(' = ')[1].startswith("CrosstableUDF") or line.split(' = ')[1].startswith("CogroupedMap") or line.split(' = ')[1].startswith("GroupedMap"):
                    start_subop = False
                if start_subop == False and line.split(' = ')[1].lstrip()[0].isupper():
                    ops.append(op_name)
        
    return pipe_code, ops
