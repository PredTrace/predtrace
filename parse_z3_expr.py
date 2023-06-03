import z3
from predicate import *
from lambda_symbolic_exec.glbs import global_uninterpreted_functions

def get_field_name_from_str(s):
    # '{}-tup-{}-{}'.format(table_name, i+1, k)
    chs = s.split('-')
    return chs[-1]

def z3_expr_requires_lambda(expr):
    if len(expr.children()) == 0: # a variable or a constant
        return False
    else:
        assert(isinstance(expr.decl(), z3.FuncDeclRef))
        children = [z3_expr_requires_lambda(e) for e in expr.children()]
        if any(children):
            return True 
        op = str(expr.decl())
        if op in ['==','>=','<=','>','<','Not','And','Or','Contains']:
            return False
        else:
            return True

# convert a z3 expr to a predicate that can be convert back to a pandas 
def convert_z3_expr(expr,use_lambda):
    if len(expr.children()) == 0: # a variable or a constant
        if isinstance(expr, z3.IntNumRef): # int const
            if use_lambda:
                return str(expr)
            else:
                return Constant(int(str(expr)))
        elif isinstance(expr, z3.ArithRef): # int variable
            if use_lambda:
                return 'xxx["{}"]'.format(get_field_name_from_str(str(expr)))
            else:
                return Field(get_field_name_from_str(str(expr)))
        elif isinstance(expr, z3.SeqRef):
            if expr.is_string_value(): # str const
                if use_lambda:
                    return str(expr)
                else:
                    return Constant(str(expr), typ='str')
            else: # str variable
                if use_lambda:
                    return 'xxx["{}"]'.format(get_field_name_from_str(str(expr)))
                else:
                    return Field(get_field_name_from_str(str(expr)))
    else:
        assert(isinstance(expr.decl(), z3.FuncDeclRef))
        children = [convert_z3_expr(e, use_lambda) for e in expr.children()]
        op = str(expr.decl())
        if op in ['==','>=','<=','>','<']:
            if use_lambda:
                return '({} {} {})'.format(children[0], op, children[1])
            else:
                return BinOp(children[0], str(op), children[1])
        elif op in ['Not']:
            if use_lambda:
                return '(!{})'.format(children[0])
            else:
                return Not(children[0])
        elif op in ['And']:
            if use_lambda:
                return '({})'.format(' & '.join(children))
            else:
                return AllAnd(*children)
        elif op in ['Or']:
            if use_lambda:
                return '({})'.format(' & '.join(children))
            else:
                return AllOr(*children)
        elif op in ['If']:
            return '({} if {} else {})'.format(children[1], children[0], children[2])
        elif op in ['StrToInt']:
            return 'int({})'.format(children[0])
        elif op in ['IntToStr']:
            return 'str({})'.format(children[0])
        elif op in ['Contains']:
            if use_lambda:
                return '({} in {})'.format(children[1], children[0])
            else:
                return BinOp(children[1], 'subset', children[0])
        elif  op in ['+','-','*','/']:
            return '({} {} {})'.format(children[0], op, children[1])
        else:
            print("UNHANDLED")
            print(expr)
            for k,v in global_uninterpreted_functions.items():
                print("(op = {}) {} : {}".format(op,k,v))
                if op == str(v):
                    return k.replace('(x)','('+children[0]+')')
            assert(False)
