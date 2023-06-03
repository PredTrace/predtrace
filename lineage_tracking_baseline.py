from interface import *

groupby_sum_code = """
def groupby_sum(df):
    ret = []
    for index,row in df.iterrows():
        ret = ret + row['lineage_tracking']
    return ret
"""

def pandas_code_with_lineage_tracking_oneop(op, output_variable_name, input_variable_names={}, table_id=0):
    if hasattr(op, 'to_python'):
        s = op.to_python(output_variable_name, input_variable_names)
        s += '\n'
    else:
        s = ''
    if isinstance(op, InitTable):
        s += "{}.reset_index(inplace=True)\n".format(output_variable_name)
        s += "{}['lineage_tracking'] = pd.Series([[({},i)] for i in range({}.shape[0])])\n".format(output_variable_name, table_id, output_variable_name)
        s += '{}.set_index("index")\n'.format(output_variable_name)
    elif isinstance(op, Filter):
        pass
    elif isinstance(op, InnerJoin):
        # each is a pair...
        s += "{}['lineage_tracking'] = {}.apply(lambda row: row['lineage_tracking_x']+row['lineage_tracking_y'], axis=1)\n".format(output_variable_name, output_variable_name)
        s += "{}.drop(columns=['lineage_tracking_x','lineage_tracking_y'], inplace=True)".format(output_variable_name)
    elif isinstance(op, LeftOuterJoin):
        s += "{}['lineage_tracking'] = {}.apply(lambda row: row['lineage_tracking_x'] if type(row['lineage_tracking_y']) is not list else (row['lineage_tracking_x']+row['lineage_tracking_y']), axis=1)\n".format(output_variable_name, output_variable_name)
        s += "{}.drop(columns=['lineage_tracking_x','lineage_tracking_y'], inplace=True)\n".format(output_variable_name)
    elif isinstance(op, GroupBy):
        s += "temp_df = {}.groupby([{}]).apply(groupby_sum).reset_index()\n".format(input_variable_names[op.dependent_ops[0]], \
            ','.join(['"{}"'.format(c) for c in op.groupby_cols]))
        s += "temp_df.columns = [{},'lineage_tracking']\n".format(','.join(['"{}"'.format(c) for c in op.groupby_cols]))
        s += "{}['lineage_tracking'] = temp_df['lineage_tracking']\n".format(output_variable_name)
    elif isinstance(op, DropDuplicate):
        pass
    elif isinstance(op, Pivot):
        s += "temp_df = {}.groupby([{}]).apply(groupby_sum).reset_index()\n".format(input_variable_names[op.dependent_ops[0]], \
            ','.join(['"{}"'.format(col) for col in op.index_col]))
        s += "temp_df.columns = [{},'lineage_tracking']\n".format(','.join(['"{}"'.format(c) for c in op.index_col]))
        s += "{}['lineage_tracking'] = temp_df['lineage_tracking']\n".format(output_variable_name)
    elif isinstance(op, SetItem):
        pass
    elif isinstance(op, SortValues):
        pass
    elif isinstance(op, DropNA):
        pass
    elif isinstance(op, FillNA):
        pass
    elif isinstance(op, Rename):
        pass
    elif isinstance(op, Copy):
        pass
    elif isinstance(op, DropColumns):
        pass
    elif isinstance(op, ChangeType):
        pass
    elif isinstance(op, Append):
        pass
    elif isinstance(op, UnPivot):
        s = "{} = {}.melt(id_vars=['lineage_tracking',{}], value_vars=[{}],var_name=\"{}\", value_name=\"{}\")\n".format(output_variable_name, input_variable_names[op.dependent_ops[0]],\
            ','.join(['"{}"'.format(c) for c in op.id_vars]),  ','.join(['"{}"'.format(c) for c in op.value_vars]), op.var_name, op.value_name)
        
    elif isinstance(op, Split):
        s += "{}['lineage_tracking'] = {}['lineage_tracking']\n".format(output_variable_name, input_variable_names[op.dependent_ops[0]])
    elif isinstance(op, TopN):
        pass
    elif isinstance(op, GetDummies):
        s += "{}['lineage_tracking'] = {}['lineage_tracking']\n".format(output_variable_name, input_variable_names[op.dependent_ops[0]])

    elif isinstance(op, AllAggregate):
        s += '{}_lineage_tracking = functools.reduce(lambda x,y: x+y["lineage_tracking"], [[]] + [row for idx,row in {}.iterrows()])\n'.format(output_variable_name, input_variable_names[op.dependent_ops[0]])
    elif isinstance(op, ScalarComputation):
        scalar_vars = []
        row_vars = []
        for k,dep_op in op.dependent_op_to_var_map.items():
            if isinstance(dep_op, AllAggregate) or isinstance(dep_op, ScalarComputation): # scalar value
                scalar_vars.append(input_variable_names[dep_op])
            else:
                row_vars.append(input_variable_names[dep_op])
        s += '{}_lineage_tracking = {} + {}\n'.format(output_variable_name, \
            '[]' if len(scalar_vars)==0 else '+'.join(['{}_lineage_tracking'.format(dep_op) for dep_op in scalar_vars]), \
            '[]' if len(row_vars)==0 else '+'.join(['{}["lineage_tracking"]'.format(dep_op) for dep_op in row_vars]))

    elif isinstance(op, SetItemWithDependency):
        variables = get_lambda_variable_names(op.apply_func)
        row_vars = []
        scalar_vars = []
        for v in variables:
            if isinstance(op.dependent_op_to_var_map[v], ScalarComputation) or isinstance(op.dependent_op_to_var_map[v], AllAggregate):
                scalar_vars.append(input_variable_names[op.dependent_op_to_var_map[v]])
            elif isinstance(op.dependent_op_to_var_map[v], SubpipeInput) and op.dependent_op_to_var_map[v].input_type == 'row': # df
                row_vars.append(input_variable_names[op.dependent_op_to_var_map[v]])
        s += '{}["lineage_tracking"] = {}["lineage_tracking"] + {} + {}\n'.format(output_variable_name, output_variable_name,\
            '[]' if len(scalar_vars)==0 else '+'.join(['{}_lineage_tracking'.format(dep_op) for dep_op in scalar_vars]), \
            '[]' if len(row_vars)==0 else '+'.join(['{}["lineage_tracking"]'.format(dep_op) for dep_op in row_vars]))

    elif isinstance(op, SubpipeInput):
        pass
    elif isinstance(op, PipelinePath):
        lines = []
        temp_input_vars = {k:v for k,v in input_variable_names.items()}
        cond_var = 'cond_{}'.format(output_variable_name)
        for i,subop in enumerate(op.operators):
            output_var = 'subpipe_{}_op_{}'.format(output_variable_name, i)
            temp_input_vars[subop] = output_var
            lines.append(pandas_code_with_lineage_tracking_oneop(subop, output_var, temp_input_vars))
        if type(op.cond) is not bool:
            lines.append(op.cond.to_python(cond_var, temp_input_vars))
        s = '\n'.join(lines) 

    elif isinstance(op, SubPipeline):
        lines = []
        if len(op.paths) == 1: # no condition
            p = op.paths[0]
            lines.append(pandas_code_with_lineage_tracking_oneop(p, '0', input_variable_names))
            if isinstance(p.operators[-1], ScalarComputation) or isinstance(p.operators[-1], AllAggregate):
                lines.append('return (subpipe_{}_op_{}, subpipe_{}_op_{}_lineage_tracking)'.format('0', len(p.operators)-1, '0', len(p.operators)-1))
            else:
                lines.append('return subpipe_{}_op_{}'.format('0', len(p.operators)-1))
        else:
            for i,p in enumerate(op.paths):
                lines.append(pandas_code_with_lineage_tracking_oneop(p, str(i), input_variable_names))
            s1 = '' # return {} if {} else ({})
            if isinstance(p.operators[-1], ScalarComputation) or isinstance(p.operators[-1], AllAggregate):
                for i,p in enumerate(op.paths):
                    if i == 0:
                        s1 = '(subpipe_{}_op_{}, subpipe_{}_op_{}_lineage_tracking)'.format(i, len(p.operators)-1, i, len(p.operators)-1)
                    else:
                        s1 = '(subpipe_{}_op_{}, subpipe_{}_op_{}_lineage_tracking) if cond_{} else {}'.format(i, len(p.operators)-1, i, len(p.operators)-1, i, s1)
            else:
                for i,p in enumerate(op.paths):
                    if i == 0:
                        s1 = 'subpipe_{}_op_{}'.format(i, len(p.operators)-1)
                    else:
                        s1 = 'subpipe_{}_op_{} if cond_{} else {}'.format(i, len(p.operators)-1, i, s1)
            lines.append("return {}".format(s1))
        s = s + '\n'.join(lines)

    elif isinstance(op, CrosstableUDF):
        func_body = pandas_code_with_lineage_tracking_oneop(op.subpipeline, '', input_variable_names)
        idx = func_body.split('\n')[0].index('=')
        input_table_name = func_body.split('\n')[0][idx+2:]

        udf_name = "udf_{}".format(hash(op)%10000)
        ret = "def {}(row, {}):\n".format(udf_name, input_table_name)
        ret += '\n'.join(['\t'+line for line in func_body.split('\n')])
        ret += '\n'
        ret += '{} = {}\n{}["{}"] = {}.apply({}, axis=1, args=[{}])\n'.format(output_variable_name, input_variable_names[op.df_to_set],\
            output_variable_name, op.new_col, output_variable_name, udf_name, input_table_name)
        ret += '{}["lineage_tracking"] = {}["{}"].apply(lambda x: x[1])\n'.format(output_variable_name, output_variable_name, op.new_col)
        ret += '{}["{}"] = {}["{}"].apply(lambda x: x[0])\n'.format(output_variable_name, op.new_col, output_variable_name, op.new_col)
        s = ret
        
    elif isinstance(op, GroupedMap):
        groupby_cols = []
        for subop in op.subpipeline.paths[0].operators:
            if isinstance(subop, SubpipeInput):
                groupby_cols = subop.group_key
        s += "temp_df = {}.groupby([{}]).apply(groupby_sum).reset_index()\n".format(input_variable_names[op.dependent_ops[0]], \
            ','.join(['"{}"'.format(c) for c in groupby_cols]))
        s += "temp_df.columns = [{},'lineage_tracking']\n".format(','.join(['"{}"'.format(c) for c in groupby_cols]))
        s += "{} = {}.merge(temp_df, left_on=[{}], right_on=[{}])\n".format(output_variable_name, output_variable_name,\
            ','.join(['"{}"'.format(col) for col in groupby_cols]), ','.join(['"{}"'.format(col) for col in groupby_cols]))

    else:
        assert(False)
    return s