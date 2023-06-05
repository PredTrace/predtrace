
import sys
sys.path.append("../../../")
sys.path.append("../../")
import z3
import dis
from interface import *
from util import *
import random

from predicate import *
from generate_input_filters import *
from compare_pushdown_result import get_output_filter, check_pushdown_result
import os

op0 = InitTable("data_0.pickle")
op2 = Filter(op0, BinOp(IsNULL(Field("date")),'==',Constant(False)))
op3 = InitTable("data_7.pickle")
op5 = Filter(op3, BinOp(IsNULL(Field("date")),'==',Constant(False)))
op6 = InitTable("data_14.pickle")
op8 = Filter(op6, BinOp(IsNULL(Field("date")),'==',Constant(False)))
op9 = InitTable("data_21.pickle")
op11 = Filter(op9, BinOp(IsNULL(Field("date")),'==',Constant(False)))
op12 = Append([op2,op5])
op13 = Append([op12,op8])
op14 = Append([op13,op11])
op15 = SortValues(op14, ["date"])
op16 = GroupBy(op15, ["date"], { "SearchFrequency":(Value(0, True),"sum") }, {"SearchFrequency":"SearchFrequency" })

ops = [op0, op2, op3, op5, op6, op8, op9, op11, op12, op13, op14, op15, op16]
output_schemas = generate_output_schemas(ops)


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
    
mkdir('./temp')


output_filter = get_output_filter(ops, './temp')

print(output_filter)
#print(output_filter)
#print(output_filter)
#exit(0)
for op_id,op_i in reversed([(k1,v1) for k1,v1 in enumerate(ops)]):
    if(op_i == ops[-1]):
        output_filter_i = {None:output_filter}
    else:
        output_filter_i = generate_output_filter_from_previous(op_i, ops)
    inference_i = op_i.get_inference_instance(output_filter_i)
    inference_i.input_filters = generate_input_filters(op_i, inference_i, AllOr(*list(output_filter_i.values())), output_schemas)
    # print(output_filter_i)
    print(op_id, ':')
    print_input_filters(inference_i)
    #print(inference_i.output_filter)

check_pushdown_result(ops, 'temp/')