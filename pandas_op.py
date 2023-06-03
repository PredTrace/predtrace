#from asyncio import new_event_loop
from tabnanny import check
from typing import final
from webbrowser import Opera
from numpy import isin
import z3
import dis
from util import *
from predicate import *
from lambda_symbolic_exec.lambda_expr_eval import *
# from constraint import *


class OperatorInference(object):

    def add_input_constraint(self, c):
        if self.input_constraint is None:
            self.input_constraint = c
        else:
            self.input_constraint = z3.And(self.input_constraint, c)
    def run_output_filter(self, output):
        ret = []
        for t in output:
            output_t = Tuple(t.values, And(t.exist_cond, self.output_filter), t.count)
            ret.append(output_t)
        return ret
    def run_input_filter(self, inputs, input_filters):
        ret = []
        for i,input_t in enumerate(inputs):
            r = []
            for t in input_t:
                cond = input_filters[i]
                new_input_t = Tuple(t.values, And(t.exist_cond, cond), t.count)
                r.append(new_input_t)
            ret.append(r)
        return ret
    def get_all_table_variables(self, include_null=False):
        vs = []
        for table in self.input_tables:
            for t in table:
                vs.extend([v1.v for k1,v1 in t.values.items()])
                if include_null:
                    vs.extend([v1.isnull for k1,v1 in t.values.items()])
        return vs

    def check_small_model(self, check_superset=False):
        return True
        
    def verify_correct(self, check_superset=False):
        rs1_temp = self.run_operator(self.input_tables)
        rs1 = self.run_output_filter(rs1_temp)
        rs2_temp = self.run_input_filter(self.input_tables, self.input_filters)
        for table in rs2_temp:
            for tup in table:
                tup.exist_cond = tup.eval_exist_cond()
        rs2 = self.run_operator(rs2_temp)
        if check_superset:
            rs2 = self.run_output_filter(rs2)
        if len(rs1) == len(rs2):
            expr = z3.And(*[rs1[i].eval_exist_cond() == rs2[i].eval_exist_cond() for i in range(len(rs1))])
            vs = self.get_all_table_variables(True)
            if self.input_constraint is None:
                return check_always_hold(expr)
            else:
                return check_always_hold(z3.Implies(self.input_constraint, expr))
        else:
            assert(False, "TODO: CANNOT VERIFY FOR DIFFERENT NUMBER OF TUPLES")

    
    def verify_lineage(self):
        rs2_temp = self.run_input_filter(self.input_tables, self.input_filters)
        for table in rs2_temp:
            for tup in table:
                tup.exist_cond = tup.eval_exist_cond()
        # if tup passes input filter, then output is not empty
        if all([len(table)==1 for table in self.input_tables]):
            rs2 = self.run_operator(rs2_temp)
            precond = []
            for table in rs2_temp:
                for tup in table:
                    precond.append(tup.exist_cond)
            precond = z3.And(*precond)
            goal = z3.Or(*[rs2[i].eval_exist_cond() for i in range(len(rs2))])
            expr = z3.Implies(precond, goal)
        elif len(self.input_tables) == 1:
            rs2 = self.run_operator([[rs2_temp[0][0]]])
            precond = self.input_tables[0][0].eval_exist_cond()
            goal = z3.Or(*[rs2[i].eval_exist_cond() for i in range(len(rs2))])
            expr = z3.Implies(precond, goal)
        else:
            exprs = []
            for i,table in enumerate(rs2_temp):
                if len(table) == 1:
                    continue
                rs2 = self.run_operator([rs2_temp[j] if j!=i else [rs2_temp[j][0]] for j in range(len(rs2_temp))])
                precond = []
                for j,tablej in enumerate(rs2_temp):
                    if j == i:
                        precond.append(tablej[0].eval_exist_cond())
                    else:
                        precond.extend([tup.eval_exist_cond() for tup in tablej])
                precond = z3.And(*precond)
                goal = z3.Or(*[rs2[j].eval_exist_cond() for j in range(len(rs2))])
                exprs.append(z3.Implies(precond, goal))
            expr = z3.And(*exprs)
        if self.input_constraint is None:
            return check_always_hold(expr)
        else:
            return check_always_hold(z3.Implies(self.input_constraint, expr))


    def output_exists(self):
        rs1_temp = self.run_operator(self.input_tables)
        rs1 = self.run_output_filter(rs1_temp)
        self.input_constraint = z3.Or(*[tup.eval_exist_cond()==True for tup in rs1])
        return self.input_constraint


class InitTableInference(OperatorInference):
    def __init__(self, input_table, output_filter):
        self.input_filters = [output_filter]
        self.output_filter = output_filter
        self.input_tables = [input_table]
        self.input_constraint = None
    def verify_correct(self, check_superset=False):
        return True
    def run_operator(self, input_tables):
        return input_tables[0]

class FilterInference(OperatorInference):
    def __init__(self, input_table, condition, output_filter):
        self.input_tables = [input_table]
        self.condition = condition
        self.output_filter = output_filter
        self.input_constraint = None
    def run_operator(self, input_tables):
        output = []
        for t in input_tables[0]:
            output_t = Tuple(t.values, zeval(And(t.exist_cond, self.condition), t), t.count)
            output.append(output_t)
        return output
        
class InnerJoinInference(OperatorInference):
    def __init__(self, table_left, table_right, cols_left, cols_right, output_filter):
        self.input_tables = [table_left, table_right]
        self.merge_cols_left = cols_left if type(cols_left) is list else [cols_left]
        self.merge_cols_right = cols_right if type(cols_right) is list else [cols_right]     
        self.output_filter = output_filter
        columns_left = set([k for k,v in table_left[0].values.items()])
        columns_right = set([k for k,v in table_right[0].values.items()])
        self.columns_used_in_left = set(get_columns_used(self.output_filter)).intersection(columns_left).difference(set(self.merge_cols_left))
        self.columns_used_in_right = set(get_columns_used(self.output_filter)).intersection(columns_right).difference(set(self.merge_cols_right))
        self.rename_left = {k:k for k,v in self.input_tables[0][0].values.items()}
        self.rename_right = {k:k for k,v in self.input_tables[1][0].values.items()}
        for col_name_l,v in self.input_tables[0][0].values.items():
            for col_name_r,v in self.input_tables[1][0].values.items():
                if col_name_l == col_name_r and col_name_l not in self.merge_cols_left and col_name_r not in self.merge_cols_right:
                   self.rename_left[col_name_l] = '{}_x'.format(col_name_l)
                   self.rename_right[col_name_r] = '{}_y'.format(col_name_r)
        self.input_constraint = None
    
    def run_operator(self, input_tables):
        table_left = input_tables[0]
        table_right = input_tables[1]
        output = []
        for left_t in table_left:
            for right_t in table_right:
                temp = {self.rename_left[k]:v for k,v in left_t.values.items()}
                temp.update( {self.rename_right[k]:v for k,v in right_t.values.items()})
                join_cond = z3.And(*[left_t[self.merge_cols_left[i]].equals(right_t[self.merge_cols_right[i]]) for i in range(len(self.merge_cols_left))]) # inner join :if merge cols left and right are different, might have bugs.
                t = Tuple(temp, z3.And(left_t.eval_exist_cond(), right_t.eval_exist_cond(), join_cond), left_t.count*right_t.count)
                output.append(t)
        return output
    def verify_correct(self, check_superset=False):
        return True

class LeftOuterJoinInference(OperatorInference):  # Yin: updated by Yin, needs to write run_opeartor
    def __init__(self, table_left, table_right, cols_left, cols_right, output_filter):
        self.input_tables = [table_left, table_right]
        self.merge_cols_left = cols_left if type(cols_left) is list else [cols_left]
        self.merge_cols_right = cols_right if type(cols_right) is list else [cols_right]     
        self.output_filter = output_filter
        columns_left = set([k for k,v in table_left[0].values.items()])
        columns_right = set([k for k,v in table_right[0].values.items()])
        self.columns_used_in_left = set(get_columns_used(self.output_filter)).intersection(columns_left).difference(set(self.merge_cols_left))
        self.columns_used_in_right = set(get_columns_used(self.output_filter)).intersection(columns_right).difference(set(self.merge_cols_right))
        self.input_constraint = None
        self.rename_left = {k:k for k,v in self.input_tables[0][0].values.items()}
        self.rename_right = {k:k for k,v in self.input_tables[1][0].values.items()}
        for col_name_l,v in self.input_tables[0][0].values.items():
            for col_name_r,v in self.input_tables[1][0].values.items():
                if col_name_l == col_name_r and col_name_l not in self.merge_cols_left and col_name_r not in self.merge_cols_right:
                   self.rename_left[col_name_l] = '{}_x'.format(col_name_l)
                   self.rename_right[col_name_r] = '{}_y'.format(col_name_r)

    def check_small_model(self, check_superset=False):
        # 𝑓(𝑓𝑗𝑜𝑖𝑛(𝑡_𝑙,𝑡_𝑟 ))=𝐹𝑎𝑙𝑠𝑒 ⋀ 𝑓𝑚𝑎𝑡𝑐ℎ(𝑡_𝑙,𝑡_𝑟 )=𝑇𝑟𝑢𝑒 → 𝐹(𝐿𝐽(𝑡_𝑙,𝑇_𝑟 ))=∅
        # one of the right row is joinable and the output of join cannot pass f, then the other rows cannot be joined/does not pass filter
        table_left = self.input_tables[0]
        table_right = self.input_tables[1]
        temp_tup1 = {self.rename_left[k]:v for k,v in table_left[0].values.items()}
        temp_tup1.update({self.rename_right[k]:v for k,v in table_right[0].values.items()})
        temp_tup2 = {self.rename_left[k]:v for k,v in table_left[0].values.items()}
        temp_tup2.update({self.rename_right[k]:v for k,v in table_right[1].values.items()})
        f1 = (zeval(self.output_filter, Tuple(temp_tup1))==False)
        f2 = z3.And(*[table_left[0][self.merge_cols_left[i]].equals(table_right[0][self.merge_cols_right[i]]) for i in range(len(self.merge_cols_left))])==True
        tup2_joinable = z3.And(*[table_left[0][self.merge_cols_left[i]].equals(table_right[1][self.merge_cols_right[i]]) for i in range(len(self.merge_cols_left))])
        fimplies = z3.Implies(tup2_joinable, zeval(self.output_filter, Tuple(temp_tup2))==False) # either not joinable, or joinable but cannot pass f
        cond1 = z3.Implies(z3.And(f1, f2), fimplies)
        
        if check_superset:
            cond1=True


        # 𝑓(𝑓𝑗𝑜𝑖𝑛(𝑡_𝑙,𝑡_𝑟 ))=𝑇𝑟𝑢𝑒 ⋀ 𝑓𝑚𝑎𝑡𝑐ℎ(𝑡_𝑙,𝑡_𝑟 )=𝑇𝑟𝑢𝑒 → 𝑔_2 (𝑡_𝑟 )=𝑇𝑟𝑢𝑒
        f1 = (zeval(self.output_filter, Tuple(temp_tup1))==True)
        fimplies = zeval(self.input_filters[1], table_right[0])==True
        cond2 = z3.Implies(z3.And(f1, f2), fimplies)

        if self.input_constraint is not None:
            return check_always_hold(z3.Implies(self.input_constraint, z3.And(cond1, cond2)))
        else:
            return check_always_hold(z3.And(cond1, cond2))

    def run_operator(self, input_tables):
        table_left = input_tables[0]
        table_right = input_tables[1]
        output = []
        right_values = set([k for k,v in table_right[0].values.items()])
        right_values = right_values - set(self.merge_cols_right)
        for left_t in table_left:
            join_conds = []
            for right_t in table_right:
                join_cond = z3.And(*[left_t[self.merge_cols_left[i]].equals(right_t[self.merge_cols_right[i]]) for i in range(len(self.merge_cols_left))])
                join_conds.append(z3.And(right_t.eval_exist_cond(), join_cond))
                temp = {self.rename_left[k]:v for k,v in left_t.values.items()}
                temp.update( {self.rename_right[k]:right_t.values[k] for k in right_values})
                t = Tuple(temp, z3.And(left_t.eval_exist_cond(), right_t.eval_exist_cond(), join_cond))
                output.append(t)
            cannot_join = z3.And(*[xx==False for xx in join_conds])
            temp = {self.rename_left[k]:v for k,v in left_t.values.items()}
            temp.update( {self.rename_right[k]:Value(table_right[0].values[k].v, True) for k in right_values})
            t = Tuple(temp, z3.And(left_t.eval_exist_cond(), cannot_join))
            output.append(t)
        return output
    

class GroupByInference(OperatorInference):
    def __init__(self, input_table, groupby_cols, aggr_func_map, new_column_names, output_filter, convert_to_tuple=None):
        self.input_tables = [input_table]
        self.groupby_cols = groupby_cols
        self.aggr_func_map = aggr_func_map #{column_name, (init_value, aggr_func)}
        self.new_column_names = new_column_names #{old_column_name, new_column_name}
        self.convert_to_tuple = convert_to_tuple # lambda 

        self.output_filter = output_filter
        self.input_constraint = None

    def verify_lineage(self):
        return True
    def run_operator(self, input_tables):
        table = input_tables[0]
        
        values = {}
        if type(self.aggr_func_map) is dict:
            for col,aggr_func_pair in self.aggr_func_map.items():
                init_value, aggr_func = aggr_func_pair
                col_name = self.new_column_names[col]
                for i in range(len(table)):
                    if col in table[0].values:
                        init_value = compute_aggregate(aggr_func, init_value, table[i].eval_exist_cond(), table[i][col])
                    else:
                        init_value = compute_aggregate_row(aggr_func, init_value, table[i].eval_exist_cond(), table[i].values)
                values[col_name] = init_value
        else:
            init_value, aggr_func = self.aggr_func_map
            for i in range(len(table)):
                init_value = compute_aggregate_row(aggr_func, init_value, table[i].eval_exist_cond(), table[i].values)
            values = self.convert_to_tuple(init_value)
        for col in self.groupby_cols:
            values[col] = table[0][col] # group value
        exist_cond = z3.Or(*[tup.eval_exist_cond() for tup in table])
        t = create_tuple(values, exist_cond, 1) # each group appear once
        output = [t]
        return output

    def check_small_model(self, check_superset=False):
        if type(self.input_filters[0]) is bool and type(self.output_filter) is bool and self.input_filters[0]==self.output_filter:
            return True
        if all([col in self.groupby_cols for col in get_columns_used(self.input_filters[0])]) and all([col in self.groupby_cols for col in get_columns_used(self.output_filter)]):
            return True

        table = self.input_tables[0]
        # prop 1: f(agg(T+U))==\empty <=> f(agg(T))==\empty and f(agg(U))==\empty
        # ------- if superset, prop 1: f(agg(T+U))==\empty & g(U)==\empty => f(agg(T))=\empty
        # prop 2: f(agg(T+U))!=\empty & g(U)==\empty => agg(T+U)==agg(T)
        # prop 3: f(agg(T))==\empty => g(T)==\empty
        if len(self.groupby_cols) == 0:
            same_group_assumption = True
        else:
            same_group_assumption = z3.And(*[z3.And(*[table[i][col_g].equals(table[0][col_g]) for col_g in self.groupby_cols]) for i in range(1, len(table))])

        t0_values = {col_g:table[0][col_g]for col_g in self.groupby_cols}
        t1_values = {col_g:table[0][col_g]for col_g in self.groupby_cols}
        t2_values = {col_g:table[0][col_g]for col_g in self.groupby_cols}
        if type(self.aggr_func_map) is dict:
            for col, aggr_func_pair in self.aggr_func_map.items():
                init_value, aggr_func = aggr_func_pair
                if aggr_func == 'count':
                    return False
                init_value_0 = init_value
                init_value_1 = init_value # T
                if col in table[0].values:
                    for i in range(len(table)):
                        init_value_0 = compute_aggregate(aggr_func, init_value_0, True, table[i][col])
                        if i != 0:
                            init_value_1 = compute_aggregate(aggr_func, init_value_1, True, table[i][col])
                    init_value_2 = compute_aggregate(aggr_func, init_value, True, table[0][col]) # U
                else:
                    for i in range(len(table)):
                        init_value_0 = compute_aggregate_row(aggr_func, init_value_0, True, table[i].values)
                        if i != 0:
                            init_value_1 = compute_aggregate_row(aggr_func, init_value_1, True, table[i].values)
                    init_value_2 = compute_aggregate_row(aggr_func, init_value, True, table[0].values) # U
                col_name = self.new_column_names[col]
                t0_values[col_name] = init_value_0 # agg(T+U)
                t1_values[col_name] = init_value_1 # agg(T)
                t2_values[col_name] = init_value_2 # agg(U), first row
                #print("agg(T+U) = {}, agg(T) = {}, agg(U) = {}".format(t0_values[col_name], t1_values[col_name], t2_values[col_name]))
        else:
            init_value, aggr_func = self.aggr_func_map
            init_value_0 = init_value
            init_value_1 = init_value # T
            if aggr_func in 'count':
                return False
            for i in range(len(table)):
                init_value_0 = compute_aggregate_row(aggr_func, init_value_0, True, table[i].values)
                if i != 0:
                    init_value_1 = compute_aggregate_row(aggr_func, init_value_1, True, table[i].values)
            init_value_2 = compute_aggregate_row(aggr_func, init_value, True, table[0].values) # U
            t0_values = self.convert_to_tuple(init_value_0)
            t1_values = self.convert_to_tuple(init_value_1)
            t2_values = self.convert_to_tuple(init_value_2)

        t0 = Tuple(t0_values)
        t1 = Tuple(t1_values)
        t2 = Tuple(t2_values)
        pre_cond_1 = z3.Implies(zeval(self.output_filter, t0)==False, zeval(self.output_filter,t2)==False)

        pre_cond_2 = []
        if type(self.aggr_func_map) is dict:
            for col, aggr_func_pair in self.aggr_func_map.items():
                col_name = self.new_column_names[col]
                pre_cond_2.append(z3.Implies(z3.And(zeval(self.output_filter,t0)==True, zeval(self.input_filters[0],table[0])==False), \
                                t0_values[col_name].equals(t1_values[col_name]) ))
        else:
            for col_name, some_v in t0_values.items():
                pre_cond_2.append(z3.Implies(z3.And(zeval(self.output_filter,t0)==True, zeval(self.input_filters[0],table[0])==False), \
                                t0_values[col_name].equals(t1_values[col_name]) ))
        if check_superset:
            pre_cond_1 = z3.Implies(z3.And(zeval(self.output_filter,t0)==False, zeval(self.input_filters[0],table[0])==False), \
                                zeval(self.output_filter, t1)==False)
        vs = self.get_all_table_variables()
        if self.input_constraint is not None:
            return check_always_hold(z3.Implies(z3.And(self.input_constraint,\
                                            same_group_assumption), z3.And(pre_cond_1, z3.And(*pre_cond_2))))
        else:
            return check_always_hold(z3.Implies(same_group_assumption, z3.And(pre_cond_1, z3.And(*pre_cond_2))))

    def verify_correct(self, check_superset=False):
        if type(self.input_filters[0]) is bool and type(self.output_filter) is bool and self.input_filters[0]==self.output_filter:
            return True
        if all([col in self.groupby_cols for col in get_columns_used(self.input_filters[0])]) and all([col in self.groupby_cols for col in get_columns_used(self.output_filter)]):
            return True

        rs1_temp = self.run_operator(self.input_tables)
        rs1 = self.run_output_filter(rs1_temp)
        rs2_temp = self.run_input_filter(self.input_tables, self.input_filters)
        rs2 = self.run_operator(rs2_temp)
        if check_superset:
            rs2 = self.run_output_filter(rs2)
        
        table = self.input_tables[0]
        if len(self.groupby_cols) == 0:
            same_group_assumption = True
        else:
            same_group_assumption = z3.And(*[z3.And([table[i][col_g].equals(table[0][col_g]) for col_g in self.groupby_cols]) for i in range(1, len(table))])

        if len(rs1) == len(rs2):
            if len(self.groupby_cols) == 0:
                expr = z3.And(*[z3.And(*[v_.equals(rs2[i][col]) for col,v_ in rs1[i].values.items()]) for i in range(len(rs1))])
            else:
                expr = z3.Implies(same_group_assumption, z3.And(*[z3.And(rs1[i].eval_exist_cond() == rs2[i].eval_exist_cond(),\
                                z3.Implies(rs1[i].eval_exist_cond(), z3.And(*[v_.equals(rs2[i][col]) for col,v_ in rs1[i].values.items()]))) for i in range(len(rs1))]))
            vs = self.get_all_table_variables()
            if self.input_constraint is not None:
                return check_always_hold(z3.Implies(self.input_constraint, expr))
            else:
                return check_always_hold(expr)
        else:
            assert(False, "TODO: CANNOT VERIFY FOR DIFFERENT NUMBER OF TUPLES")

    def verify_lineage(self):
        if len(self.groupby_cols) > 0:
            table = self.input_tables[0]
            same_group_assumption = z3.And(*[z3.And([table[i][col_g].equals(table[0][col_g]) for col_g in self.groupby_cols]) for i in range(1, len(table))])
            if self.input_constraint is None:
                self.input_constraint = same_group_assumption
            else:
                self.input_constraint = z3.And(self.input_constraint, same_group_assumption)
        else:
            self.input_constraint = True
        rs2_temp = self.run_input_filter(self.input_tables, self.input_filters)
        exprs = []
        for row in rs2_temp[0]:
            rs = self.run_operator([[row]])
            exprs.append(rs[0].eval_exist_cond())
        return check_always_hold(z3.Implies(self.input_constraint, z3.And(*exprs)))


class DropDuplicateInference(GroupByInference):
    def __init__(self, input_table, cols, output_filter):
        self.cols = cols
        input_columns = [k for k,v in input_table[0].values.items()]
        other_columns = set(input_columns).difference(set(cols))
        aggr_func_map = {c: (input_table[0][c], 'lambda x,y: y') for c in other_columns}
        new_column_names = {c:c for c in other_columns}
        super().__init__(input_table, cols, aggr_func_map, new_column_names, output_filter)

class RenameInference(OperatorInference):
    def __init__(self, input_table, rename_columns, output_filter):
        self.input_tables = [input_table]
        self.rename_columns = rename_columns
        self.output_filter = output_filter
        reversed_rename = {}
        for k,v in self.rename_columns.items():
            reversed_rename[v] = k
        self.input_constraint = None
    def verify_lineage(self):
        return True
    def verify_correct(self, check_superset=False):
        return True
    def run_operator(self, input_tables):
        ret = []
        for tup in input_tables[0]:
            new_tup = Tuple({(k1 if k1 not in self.rename_columns else self.rename_columns[k1]):v1 \
                for k1,v1 in tup.values.items()}, get_filter_replacing_field(tup.exist_cond, self.rename_columns))
            ret.append(new_tup)
        return ret

class PivotInference(OperatorInference): 
    def __init__(self, input_table, index_col, header_col, value_col, value_aggr_func, output_filter, output_schema):
        self.input_tables = [input_table]
        self.index_col = index_col
        self.header_col = header_col
        self.value_col = value_col
        self.value_aggr_func = value_aggr_func 
        
        self.output_filter = output_filter
        self.output_schema = output_schema 

        if value_aggr_func is None:
            init_value = {k:Value(0, True) for k,v in self.output_schema.items()}
            lambda_expr = "lambda x, row: {{k:(row['{}'] if k == row['{}'] else v) for k,v in x.items()}}".format(self.value_col, self.header_col)
            convert_to_tuple = "lambda x: x"
            self.groupby_inference = GroupByInference(input_table, [self.index_col], (init_value, lambda_expr), {}, output_filter, convert_to_tuple)
        self.input_constraint = None

    def run_operator(self, input_tables):
        return self.groupby_inference.run_operators(input_tables)
    def check_small_model(self, check_superset=False):
        # return True
        return self.groupby_inference.check_small_model()
    def verify_correct(self, check_superset=False):
        # return True
        return self.groupby_inference.verify_correct()


class SetItemInference(OperatorInference): # same as point-wise apply
    def __init__(self, input_table, new_col, apply_func, return_type, output_filter):
        self.input_tables = [input_table]
        self.new_col = new_col
        self.apply_func = apply_func
        self.return_type = return_type
       
        self.output_filter = output_filter
        self.input_constraint = None

    def verify_lineage(self):
       return True
    def verify_correct(self):
        return True
    def run_operator(self, input_tables):
        table = input_tables[0]
        ret = []
        involved_cols = get_columns_used(self.apply_func)
        for tup in table:
            newv = eval_lambda(self.apply_func, self.return_type, tup.values).v 
            value_copy = {k:v for k,v in tup.values.items()}
            value_copy[self.new_col] = Value(newv, False) 
            ret.append(Tuple(value_copy, tup.eval_exist_cond(), tup.count)) 
        return ret
            
    
class SortValuesInference(OperatorInference):
    def __init__(self, input_table, cols, output_filter):
        self.input_tables = [input_table]
        self.output_filter = output_filter
        self.cols = cols
        self.input_constraint = None
    def verify_lineage(self):
        return True
    def run_operator(self, input_tables):
        return input_tables[0]


# # ILoc cannot be push down

class UnpivotInference(OperatorInference): 
    def __init__(self, input_table, id_vars, value_vars, var_name, value_name, output_filter):
        self.input_tables = [input_table]
        self.output_filter = output_filter
        self.id_vars = id_vars
        self.value_vars = value_vars
        self.var_name = var_name
        self.value_name = value_name
        self.input_constraint = None
    def run_operator(self, input_tables):
        ret = []
        for tup in input_tables[0]:
            for i,value_var in enumerate(self.value_vars):
                tup_values = {id_var:tup[id_var] for id_var in self.id_vars}
                tup_values[self.var_name] = Constant(value_var)
                tup_values[self.value_name] = tup[value_var]
                ret.append(Tuple(tup_values, tup.eval_exist_cond()))
        return ret
    def verify_correct(self):
        return True
    

class SplitInference(OperatorInference): # 
    def __init__(self, input_table, column_to_split, new_col_names, regex, by, output_filter):
        self.input_tables = [input_table]
        self.output_filter = output_filter
        self.column_to_split = column_to_split
        self.regex = regex
        self.new_col_names = new_col_names
        self.by = by
        self.input_constraint = None
        self.set_item_inferences = []
    def verify_lineage(self):
        return True
    def run_operator(self, input_tables):
        table = input_tables[0]
        new_table = []
        for tup in table:
            new_tup = {k:v for k,v in tup.values.items()}
            del new_tup[self.column_to_split]
            if self.regex is not None:
                for i,new_col in enumerate(self.new_col_names):
                    lambda_str = 'lambda x:list(filter(None, re.split(r"{}", x["{}"])))[{}]'.format(self.regex, self.column_to_split, i)
                    new_tup[new_col] = eval_lambda(lambda_str,'str',tup.values)
            elif self.by is not None:
                for i,new_col in enumerate(self.new_col_names):
                    lambda_str = 'lambda x:x["{}"].split("{}")[{}]'.format(self.column_to_split, self.by, i)
                    new_tup[new_col] = eval_lambda(lambda_str,'str',tup.values)
            else:   
                assert(False)
            new_table.append(Tuple(new_tup, tup.exist_cond))
        return new_table

class CopyInference(OperatorInference): # do nothing
    def __init__(self, input_table, output_filter):
        self.input_tables = [input_table]
        self.output_filter = output_filter
        self.input_constraint = None
    def verify_lineage(self):
        return True
    def run_operator(self, input_tables):
        return input_tables[0]


class DropColumnsInference(OperatorInference): # Yin: TODO: add run_operator, verify correct and so on to new operators 
    def __init__(self, input_table, cols, output_filter):
        self.input_tables = [input_table]
        self.output_filter = output_filter
        self.cols = cols
        self.input_constraint = None
    def verify_lineage(self):
        return True
    def verify_correct(self):
        return True
    def run_operator(self, input_tables):
        new_table = []
        for tup in input_tables[0]:
            new_tup = {k:v for k,v in tup.values.items()}
            for col in self.cols:
                del new_tup[col]
            new_table.append(Tuple(new_tup, tup.exist_cond))
        return new_table

class AppendInference(OperatorInference):
    def __init__(self, input_tables, output_filter):
        self.input_tables = input_tables
        self.output_filter = output_filter
        self.input_constraint = None
    def verify_lineage(self):
        return True
    def run_operator(self, input_tables):
        new_table = []
        for table in input_tables:
            for tup in table:
                new_table.append(tup)
        return new_table
    def verify_correct(self, check_superset=False):
        return True

class ConcatColumnInference(OperatorInference):
    def __init__(self, input_tables, output_filter):
        self.input_tables = input_tables
        self.output_filter = output_filter
        self.input_constraint = None
    def run_operator(self, input_tables):
        new_table = []
        for i in range(len(input_tables[0])):
            new_tup = {}
            for tup in input_tables:
                new_tup.update({k:v for k,v in tup.values.items()})
            new_table.append(Tuple(new_tup))
        return new_table
    def verify_correct(self, check_superset=False):
        return False



class GetDummiesInference(OperatorInference): 
    def __init__(self, input_table, cols, name_map, output_filter):
        self.input_tables = [input_table]
        self.cols = cols
        self.name_map = name_map
        self.output_filter = output_filter
        self.input_constraint = None
    def verify_lineage(self):
        return True
    def verify_correct(self):
        return True

class ChangeTypeInference(SetItemInference):
    def __init__(self, input_table, target_type, orig_col, new_col, output_filter):
        if target_type in ['date','datetime']:
            apply_func = 'lambda x: pd.to_datetime(x["{}"])'.format(orig_col)
        elif target_type in ['str']:
            apply_func = 'lambda x: str(x["{}"])'.format(orig_col)
        elif target_type in ['float']:
            apply_func = 'lambda x: float(x["{}"])'.format(orig_col)
        else:
            apply_func = 'lambda x: int(x["{}"])'.format(orig_col)
        super().__init__(input_table, new_col, apply_func, target_type, output_filter)
 
class DropNAInference(FilterInference):
    def __init__(self, input_table, cols, output_filter):
        condition = AllOr(*[IsNotNULL(Field(col)) for col in cols])  # how = 'any'
        super().__init__(input_table, condition, output_filter)
    
        
class FillNAInference(SetItemInference):
    def __init__(self, input_table, col, fill_value, output_filter):
        # TODO: get the column used from lambda
        self.input_tables = [input_table]
        self.fill_value = fill_value
        # Yin :self.f - [lambda function, fill_value]
        self.f = 'lambda x: {} if pd.isnull(x["{}"]) else x["{}"]'.format(self.fill_value, col, col)
        #self.f = lambda x: fill_value if pd.isnull(x[col]) else x[col]
        #lambda x: z3.If(x[col].isnull, fill_value, x[col].v)
        super().__init__(input_table, col, self.f, get_variable_type(self.fill_value), output_filter)
    def verify_correct(self, check_superset=False):
        return True

class TopNInference(OperatorInference):
    def __init__(self, input_table, N, sort_order, desc, output_filter):
        self.input_tables = [input_table]
        self.N = N
        self.sort_order = sort_order
        self.desc = desc
        self.output_filter = output_filter
        self.input_constraint = None
    def verify_lineage(self):
        return True
    def check_small_model(self, check_superset=False):
        if check_superset:
            return True
        assert(len(self.input_tables[0])>= self.N)
        table = self.input_tables[0]
        cur_count = 0
        for tup in self.input_tables[0][:self.N]:
            exist_cond = z3.And(zeval(self.output_filter, tup), cur_count<self.N)
            cur_count = z3.If(exist_cond, cur_count+1, cur_count)
        assump1 = z3.And(cur_count>0, cur_count < self.N)
        
        if self.desc:
            descending_constr = True
            for i in range(len(table)-1):
                descending_constr = z3.And(descending_constr, table[i][self.sort_order[0]].v>=table[i+1][self.sort_order[0]].v)
            newtup_values = {}
            for k,v in table[0].values.items():
                newtup_values[k] = Value(get_new_variable_by_type(get_variable_type(v.v),' other-{}'.format(k)))
            minv = table[0][self.sort_order[0]].v 
            for row in table[1:]:
                minv = z3.If(minv<row[self.sort_order[0]].v, minv, row[self.sort_order[0]].v)
            target = zeval(self.output_filter, Tuple(newtup_values))==False
            expr = z3.Implies(z3.And(descending_constr, newtup_values[self.sort_order[0]].v<=minv, assump1), target)
        else:
            descending_constr = True
            for i in range(len(table)-1):
                descending_constr = z3.And(descending_constr, table[i][self.sort_order[0]].v<=table[i+1][self.sort_order[0]].v)
            newtup_values = {}
            for k,v in table[0].values.items():
                newtup_values[k] = Value(get_new_variable_by_type(get_variable_type(v.v),' other-{}'.format(k)))
            maxv = table[0][self.sort_order[0]].v
            for row in table[1:]:
                maxv = z3.If(maxv>row[self.sort_order[0]].v, maxv, row[self.sort_order[0]].v)
            target = zeval(self.output_filter, Tuple(newtup_values))==False
            expr = z3.Implies(z3.And(descending_constr, newtup_values[self.sort_order[0]].v>=maxv, assump1), target)
            print("expr = {}".format(z3.simplify(target)))

        vs = self.get_all_table_variables()+[newtup_values[self.sort_order[0]].v]
        if self.input_constraint is not None:
            return check_always_hold(z3.Implies(self.input_constraint, expr))
        else:
            return check_always_hold(expr)

    def run_operator(self, input_tables):
        new_table = []
        cur_count = 0
        for tup in input_tables[0]:
            new_tup = {k:v for k,v in tup.values.items()}
            exist_cond = z3.And(tup.eval_exist_cond(), cur_count<self.N)
            cur_count = z3.If(exist_cond, cur_count+1, cur_count)
            new_table.append(Tuple(new_tup, exist_cond))
        return new_table
    
    def verify_correct(self, check_superset=False):
        table = self.input_tables[0]
        if self.desc:
            descending_constr = True
            for i in range(len(table)-1):
                descending_constr = z3.And(descending_constr, table[i][self.sort_order[0]].v>=table[i+1][self.sort_order[0]].v)
        else:
            descending_constr = True
            for i in range(len(table)-1):
                descending_constr = z3.And(descending_constr, table[i][self.sort_order[0]].v<=table[i+1][self.sort_order[0]].v)
        
        if self.input_constraint is not None:
            self.input_constraint = z3.And(self.input_constraint, descending_constr)
        else:
            self.input_constraint = descending_constr
        return super().verify_correct(check_superset)
    def verify_lineage(self):
        table = self.input_tables
        if self.desc:
            descending_constr = True
            for i in range(len(table)-1):
                descending_constr = z3.And(descending_constr, table[i][self.sort_order[0]].v>=table[i+1][self.sort_order[0]].v)
        else:
            descending_constr = True
            for i in range(len(table)-1):
                descending_constr = z3.And(descending_constr, table[i][self.sort_order[0]].v<=table[i+1][self.sort_order[0]].v)
        if self.input_constraint is not None:
            self.input_constraint = z3.And(self.input_constraint, descending_constr)
        else:
            self.input_constraint = descending_constr
        return super().verify_lineage()
    

class AllAggrInference(GroupByInference):
    def __init__(self, input_table, initv, aggr_func, output_filter, convert_to_tuple=None):
        if convert_to_tuple is None:
            super().__init__(input_table, [], (initv, aggr_func), {}, output_filter, lambda x: {'new_aggr_col':x})
        else:
            super().__init__(input_table, [], (initv, aggr_func), {}, output_filter, convert_to_tuple)
    
class SetItemWithDependencyInference(SetItemInference):
    def __init__(self, input_table, df_to_set, new_col, dependent_op_to_var_map, return_type, apply_func, output_filter):
        self.input_tables = [input_table]
        self.df_to_set = df_to_set
        self.new_col = new_col
        self.dependent_op_to_var_map = dependent_op_to_var_map
        self.apply_func = apply_func
        self.return_type = return_type
        self.output_filter = output_filter
    def run_operator(self, input_tables):
        table = input_tables[0]
        ret = []
        variables = get_lambda_variable_names(self.apply_func)
        for tup in table:
            passed_variables = [z3.Int('v-{}'.format(v)) if self.dependent_op_to_var_map[v]!=self.df_to_set else tup.values for v in variables]
            newv = eval_lambda_other(self.apply_func, self.return_type, *passed_variables) 
            value_copy = {k:v for k,v in tup.values.items()}
            value_copy[self.new_col] = Value(newv, False) 
            ret.append(Tuple(value_copy, tup.eval_exist_cond(), tup.count))  
        return ret

class SubpipeInputInference(OperatorInference):
    def __init__(self, input_table, input_type, group_key, output_filter):
        self.input_type = input_type
        self.group_key = group_key
        self.output_filter = output_filter
        self.input_tables = [input_table]
    def run_operator(self, input_tables):
        return self.input_tables[0] 

class ScalarComputationInference(OperatorInference):
    def __init__(self, input_tables, expr, output_filter):
        self.input_tables = input_tables
        self.output_filter = output_filter
        self.expr = expr
        self.input_constraint = None
        self.output_tup = None
    def verify_lineage(self):
        return True
    def run_operator(self, input_tables): # input_tables is a map
        input_exists = []
        input_values = []
        for v in input_tables:
            if isinstance(v, Tuple):
                input_exists.append(v.eval_exist_cond())
                input_values.append(v.values)
            elif isinstance(v, Value):
                input_exists.append(True)
                input_values.append(v.v)
            elif type(v) is list:
                if len(v[0].values) and 'new_aggr_col' in v[0].values: # input is AllAggr, turn into a single input value
                    input_values.append(v[0].values['new_aggr_col'].v)
                    input_values.append(True)
                else:
                    input_values.append(v[0])
                    input_exists.append(v[0].eval_exist_cond())
        input_exists = z3.And(*input_exists)
        variable_instances = input_values
        expr = eval_lambda_other(self.expr, None, *variable_instances).v
        self.output_tup = Tuple({'scalar_out':Value(expr)}, input_exists)
        return self.output_tup
    def eval(self, tup):
        if self.output_tup is None:
          self.run_operator(self.input_tables)
        return self.output_tup.values['scalar_out'].v
    def verify_correct(self, check_superset=False):
        out_tup = self.run_operator(self.input_tables)
        if check_superset:
            expr = z3.Implies(zeval(self.output_filter, out_tup), out_tup.input_exists)
        else:
            expr = zeval(self.output_filter, out_tup) == out_tup.input_exists
        if self.input_constraint is not None:
            return check_always_hold(z3.Implies(self.input_constraint, expr))
        else:
            return check_always_hold(expr)
    def verify_lineage(self):
        input_tables = self.run_input_filter(self.input_tables, self.input_filters)
        exprs = []
        for v in input_tables:
            input_exists = []
            if isinstance(v, Tuple):
                input_exists.append(v.eval_exist_cond())
            elif isinstance(v, Value):
                input_exists.append(True)
            elif type(v) is list:
                if len(v[0].values)>0 and 'new_aggr_col' in v[0].values: # input is AllAggr, turn into a single input value
                    input_exists.append(True)
                else:
                    input_exists.append(v[0].eval_exist_cond())
            exprs.append(z3.And(*input_exists))
        expr = z3.Or(*exprs)
        if self.input_constraint is not None:
            return check_always_hold(z3.Implies(self.input_constraint, expr))
        else:
            return check_always_hold(expr)

class CrosstableUDFOnePathInference(OperatorInference):
    def __init__(self, input_table, new_col, sub_pipeline, sub_pipeline_dependency, path_cond, output_filter):
        self.input_tables = [input_table]
        self.new_col = new_col
        self.sub_pipeline = sub_pipeline
        self.sub_pipeline_dependency = sub_pipeline_dependency 
        self.path_cond = path_cond
        self.input_constraint = None 
        self.output_filter = output_filter
        self.input_filters = []
        # row + table -> one_value

    def run_operator(self, input_tables):
        ret = []
        for tup in input_tables[0]:
            out_value_map = {} 
            for i,op in enumerate(self.sub_pipeline):
                print(type(op))

                inputs = []
                if isinstance(op, SubpipeInputInference):
                    if op.input_type == 'row':
                        out_value_map[i] = tup
                    else:
                        out_value_map[i] = input_tables[1]
                else:
                    for dep_id in self.sub_pipeline_dependency[i]:
                        inputs.append(out_value_map[dep_id])
                    output = op.run_operator(inputs)
                    out_value_map[i] = output
            values = {k:v for k,v in tup.values.items()}
            last_op_idx = len(self.sub_pipeline)-1 if type(self.path_cond) is bool else len(self.sub_pipeline)-2
            final_out = out_value_map[last_op_idx]
            final_exist = True
            if type(final_out) is list:
                final_out_v = [v for k,v in final_out[0].values.items()][0].v
                final_exist = final_out[0].eval_exist_cond()
            elif type(final_out) is Tuple:
                final_out_v = final_out.values['scalar_out'].v
                final_exist = final_out.eval_exist_cond()
            values[self.new_col] = Value(final_out_v)
            ret.append(Tuple(values, tup.eval_exist_cond()))

        if not type(self.path_cond) is bool:
            path_cond = out_value_map[len(self.sub_pipeline)-1].values['scalar_out'].v
        else:
            path_cond = True
        return ret, path_cond
    
    def output_exists(self):
        table_right_op = None
        for op in self.sub_pipeline:
            if isinstance(op, SubpipeInputInference) and op.input_type == 'table':
                table_right_op = op
        
        final_output,path_cond1 = self.run_operator(self.input_tables+table_right_op.input_tables)
        self.input_constraint = zeval(self.output_filter, final_output[0])==True
        return self.input_constraint

    def verify_correct(self, check_superset=False):
        table_right_op = None
        for op in self.sub_pipeline:
            if isinstance(op, SubpipeInputInference) and op.input_type == 'table':
                table_right_op = op
        left_eval = zeval(self.input_filters[0], self.input_tables[0][0])
        final_output,path_cond1 = self.run_operator(self.input_tables+table_right_op.input_tables)
        final_output_exist = zeval(self.output_filter, final_output[0])
        # ∀𝑡_𝑙,𝑡_𝑟, 𝑔_l (𝑡_𝑙)=𝐹𝑎𝑙𝑠𝑒 → 𝑓(𝑂𝑝(𝑡_𝑙,𝑡_𝑟 ))=𝐹𝑎𝑙𝑠𝑒
        assumption1 = z3.Implies(left_eval==False, final_output_exist==False)

        # ∀𝑡_𝑙, 𝑇_𝑟,𝑔_l (𝑡_𝑙)=𝑇𝑟𝑢𝑒 → 𝑓(𝑂𝑝(𝑡_𝑙,𝑇_𝑟 ))=𝑂𝑝(𝑡_𝑙, 𝑔_r (𝑇_𝑟 )) 
        assumption2 = []
        rs2_temp = self.run_input_filter(self.input_tables+table_right_op.input_tables, self.input_filters)
        output_with_input_filter,path_cond2 = self.run_operator(rs2_temp)
        if check_superset:
            output_with_input_filter = self.run_output_filter(output_with_input_filter)
        # path condition same
        assumption0 = z3.Implies(left_eval==True, (path_cond1==path_cond2))
        
        
        for i in range(len(final_output)):
            assumption2.append(z3.And(final_output_exist == output_with_input_filter[i].eval_exist_cond(),\
            z3.Implies(final_output_exist==True, final_output[i].values[self.new_col].v==output_with_input_filter[i].values[self.new_col].v)))
        assumption2 = z3.Implies(left_eval==True, z3.And(*assumption2))

        vs = self.get_all_table_variables()
        vs = vs + table_right_op.get_all_table_variables()
        if self.input_constraint is not None:
            self.input_constraint = z3.And(self.input_constraint, path_cond1==True)
        else:
            self.input_constraint = (path_cond1==True)

        return check_always_hold(z3.Implies(self.input_constraint, z3.And(assumption0, assumption1, assumption2)), vs)
        

class CrosstableUDFInference(OperatorInference):
    def __init__(self, inferences, output_filter):
        self.inferences = inferences
        self.output_filter = output_filter
    def verify_correct(self, check_superset=False):
        return True

        
class GroupedMapOnePathInference(OperatorInference):
    def __init__(self, input_table, group_cols, sub_pipeline, sub_pipeline_dependency, path_cond, output_filter):
        self.input_tables = [input_table]
        self.sub_pipeline = sub_pipeline
        self.group_cols = group_cols
        self.sub_pipeline_dependency = sub_pipeline_dependency # key: sub_pipeline_idx, value: [idxes]
        self.path_cond = path_cond
        self.input_constraint = z3.And(*[z3.And(*[row[col].v==input_table[0][col].v for col in group_cols]) for row in input_table[1:]])
        self.output_filter = output_filter
        self.input_filters = []
        self.output_type = 'table'
        if isinstance(self.sub_pipeline[-1], ScalarComputationInference) or isinstance(self.sub_pipeline[-1], AllAggrInference):
            self.output_type = 'scalar'

    def run_operator(self, input_tables):
        ret = []
        out_value_map = {}
        for i,op in enumerate(self.sub_pipeline):
            #print(type(op))
            inputs = []
            if isinstance(op, SubpipeInputInference):
                out_value_map[i] = input_tables[0]
            else:
                for dep_id in self.sub_pipeline_dependency[i]:
                    inputs.append(out_value_map[dep_id])
                output = op.run_operator(inputs)
                out_value_map[i] = output
        last_op_idx = len(self.sub_pipeline)-1 if type(self.path_cond) is bool else len(self.sub_pipeline)-2
        final_out = out_value_map[last_op_idx]
        if self.output_type == 'table':
            ret = final_out
        else : # scalara output, df -> value
            final_exist = True
            if type(final_out) is list: # AllAggr
                final_out_v = [v for k,v in final_out[0].values.items()][0].v
                final_exist = final_out[0].eval_exist_cond()
            elif type(final_out) is Tuple: # Scalar
                final_out_v = final_out.values['scalar_out'].v
                final_exist = final_out.eval_exist_cond()
            #print("--------FINAL RETURN VALUE = {}".format(z3_simplify(final_out_v)))
            #print('--------final exist = {}'.format(z3_simplify(final_exist)))
            ret.append(Tuple({'output':final_out_v}, final_exist))

        if not type(self.path_cond) is bool:
            path_cond = out_value_map[len(self.sub_pipeline)-1].values['scalar_out'].v
        else:
            path_cond = True
        return ret, path_cond
    
    def check_small_model(self, check_superset=False):
        return z3.And(*[inf.check_small_model(check_superset) for inf in self.sub_pipeline])

    def verify_correct(self, check_superset=False):
        for op in self.sub_pipeline:
            if len(self.input_filters) == 0 and isinstance(op, SubpipeInputInference) and len(op.input_filters) > 0:
                self.input_filters = op.input_filters
        rs1,path_cond1 = self.run_operator(self.input_tables)
        rs1 = self.run_output_filter(rs1)
        print("===========")
        rs2_temp = self.run_input_filter(self.input_tables, self.input_filters)
        rs2,path_cond2 = self.run_operator(rs2_temp)
        if check_superset:
            rs2 = self.run_output_filter(rs2)

        assumption0 = (path_cond1==path_cond2)
        assumption1 = z3.And(*[z3.And(rs1[i].eval_exist_cond()==rs2[i].eval_exist_cond(), \
            z3.And(*[(str(rs1[i][col].v) == str(rs1[i][col].v) or rs1[i][col].v==rs2[i][col].v) for col,x in rs1[i].values.items()])) for i in range(len(rs1))])
        
        vs = self.get_all_table_variables()
        if self.input_constraint is not None:
            self.input_constraint = z3.And(self.input_constraint, path_cond1==True)
        else:
            self.input_constraint = (path_cond1==True)

        return check_always_hold(z3.Implies(self.input_constraint, z3.And(assumption0, assumption1)), vs)

class GroupedMapInference(OperatorInference):
    def __init__(self, inferences, output_filter):
        self.inferences = inferences
        self.output_filter = output_filter
    def verify_correct(self, check_superset=False):
        x = []
        for i in self.inferences:
            x.append(i.verify_correct(check_superset))
        return all(x)