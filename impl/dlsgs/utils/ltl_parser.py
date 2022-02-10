#!python3.6

# implementation based on DeepLTL https://github.com/reactive-systems/deepltl

# pylint: disable=line-too-long

from enum import Enum
import re
import random
from functools import reduce, total_ordering
import operator

import sympy as sy
import sympy.logic as syl

# Regular expression for allowed APs. Can disallow 't' and 'f', but all lowercase characters should be fine
ap_alphabetic_re = r'[a-z]'
# ap_alphabetic_re = '[a-eg-su-z]'
ap_p_numeric_re = r'p\d+'


def ltl_formula(formula_string: str, format: str = 'spot') -> 'LTLFormula':
    """Parse a LTL formula in the specified format (spot, lbt, network-infix, network-polish)"""
    token_list = tokenize_formula(formula_string, format.split('-')[0])
    if format == 'spot' or format == 'network-infix':
        tree, remainder = parse_infix_formula(token_list)
    elif format == 'lbt' or format == 'network-polish':
        tree, remainder = parse_polish_formula(token_list)
    else:
        raise ValueError("'format' must be one of: spot, lbt, network-infix, network-polish")
    if remainder:
        raise ParseError("Could not fully parse formula, remainder: '" + str(remainder) + "'")
    return tree


class LTLFormula():
    """Represents a parsed LTL formula, use to_str() to get a representation in the desired format (spot, lbt, network-infix, network-polish)"""

    def __str__(self):
        return self.to_str(format='spot')

    def to_str(self, format='spot', spacing=None, full_parens=False) -> str:  # spacing: 'none' (a&X!b), 'binary ops' (a & X!b), 'all ops' (a & X ! a)
        if format == 'spot':
            return self._to_str('infix', 'spot', spacing=spacing if spacing is not None else 'binary ops', full_parens=full_parens)
        elif format == 'lbt':
            return self._to_str('polish', 'lbt', spacing=spacing if spacing is not None else 'all ops', full_parens=full_parens)
        elif format == 'network-infix':
            if spacing is None:
                spacing = 'none'
            return self._to_str('infix', 'network', spacing=spacing, full_parens=full_parens)
        elif format == 'network-polish':
            if spacing is None:
                spacing = 'none'
            return self._to_str('polish', 'network', spacing=spacing, full_parens=full_parens)
        else:
            raise ValueError("Unrecognized format")

    def binary_position_list(self, format, add_first):
        """Returns a list of binary lists where each list represents the position of a node in the abstract syntax tree as a sequence of steps along tree branches starting in the root node with each step going left ([1,0]) or going right([0,1]). The ordering of the lists is given by the format. add_first specifies whether branching choices are added at the first position or at the last position of a list.
        Example: Given the formula aUb&Xc depeding on the parameters the following lists are returned.
            format=spot, add_first=True: [[1,0,1,0],[1,0],[0,1,1,0],[],[0,1],[1,0,0,1]]
            format=spot, add_first=False: [[1,0,1,0],[1,0],[1,0,0,1],[],[0,1],[0,1,1,0]]
            format=lbt, add_first=True: [[], [1,0], [1,0,1,0], [0,1,1,0], [0,1], [1,0,0,1]]
            format=lbt, add_first=False: [[],[1,0],[1,0,1,0],[1,0,0,1],[0,1],[0,1,1,0]]
        """
        raise NotImplementedError()

    def _to_str(self, notation, format_, spacing, full_parens):
        raise NotImplementedError()

    def equal_to(self, other: 'LTLFormula', extended_eq=False):
        raise NotImplementedError()

    def size(self):
        raise NotImplementedError()

    def contained_aps(self):
        raise NotImplementedError()

    def rewrite(self, token):
        raise NotImplementedError()

    def __add__(self, other):
        if other is None:
            return self
        if not isinstance(other, LTLFormula):
            raise ValueError('and operand is no formula')
        return LTLFormulaBinaryOp(Token.AND, self, other)

    def __radd__(self, other):
        return self.__add__(other)

    def to_sympy(self):
        raise NotImplementedError()

    def negation_normal_form(self, negate=False):
        raise NotImplementedError()

    def relax_to_prop(self):
        raise NotImplementedError()


def ltl_trace(trace_string, format, symbolic=True, ap_format='alphabetic'):
    """Parse a LTL trace in the specified format (spot, lbt, network-infix, network-polish)"""
    if format.startswith('network'):
        if ap_format == 'alphabetic':
            ap_re = ap_alphabetic_re
        elif ap_format == 'p_numeric':
            ap_re = ap_p_numeric_re
        else:
            raise ValueError()
        if symbolic:
            network_step_re = '([&|!10]|' + ap_re + ')+'
        else:
            network_step_re = '(1|(!?' + ap_re + ',)*!?' + ap_re + ')'
        network_output_re = '0|' + '(' + network_step_re + ';)*' + '[{]' + '(' + network_step_re + ';)*' + network_step_re + '[}]'
        if not re.fullmatch(network_output_re, trace_string):
            raise ParseError("Network output did not match regex")

    m = re.fullmatch('(.*)' + ('cycle' if not format.startswith('network') else '') + '[{](.+)[}]', trace_string)
    if m is None:
        raise ParseError("Could not match cycle part of trace")
    prefix_steps = m.groups()[0].split(';')
    cycle_steps = m.groups()[1].split(';')
    prefix_steps = prefix_steps[:-1]  # remove element past last ;
    if symbolic:
        prefix_formulas = [ltl_formula(step_string, format=format) for step_string in prefix_steps]
        cycle_formulas = [ltl_formula(step_string, format=format) for step_string in cycle_steps]
    else:
        prefix_formulas = [LiteralSet.from_str(step_string, format=format) for step_string in prefix_steps]
        cycle_formulas = [LiteralSet.from_str(step_string, format=format) for step_string in cycle_steps]
    return LTLTrace(prefix_formulas, cycle_formulas, symbolic=symbolic)

@total_ordering
class Literal:
    def __init__(self, name, negated=False):
        self.name = name
        self.negated = negated
    def to_str(self, **kwargs):
        return ('!' if self.negated else '') + self.name
    def contained_aps(self):
        if self.name == '1':
            return set()
        else:
            return {self.name}

    def __hash__(self):
        return hash((self.name, self.negated))

    def __eq__(self, other: 'Literal'):
        return self.negated == other.negated and self.name == other.name

    def __lt__(self, other: 'Literal'):
        return self.name < other.name

    def to_formula(self, boolean=True):
        if boolean:
            f = F_AP(self.name)
            return F_NOT(f) if self.negated else f
        else:
            raise ValueError
    


class LiteralSet:
    def __init__(self, literals):
        self.literals = set(literals)

    @classmethod
    def from_str(cls, s, **kwargs):
        s = s.split(',')
        literals = []
        for lit in s:
            if lit[0] == '!':
                literals.append(Literal(lit[1:], True))
            else:
                literals.append(Literal(lit, False))
        return cls(literals)

    def to_str(self, *args, **kwargs):
        return ','.join([q.to_str() for q in self.literals])

    def equal_to(self, other: 'LiteralSet', **kwargs):
        return len(self.literals) == len(other.literals) and all([a == b for a, b in zip(sorted(self.literals), sorted(other.literals))])

    def contained_aps(self):
        return reduce(operator.or_, [q.contained_aps() for q in self.literals])

    def to_formula(self, boolean=True):
        return reduce(F_AND, (q.to_formula(boolean=boolean) for q in self.literals))


class LTLTrace():
    """Represents a parsed LTL trace, use to_str() to get a representation in the desired format (spot, network-infix, network-polish)"""

    def __init__(self, prefix_formulas, cycle_formulas, symbolic=False):
        self.prefix_formulas = prefix_formulas
        self.cycle_formulas = cycle_formulas
        self.symbolic = symbolic

    def to_str(self, format):
        space = '' if format.startswith('network') else ' '
        cycle_start = '{' if format.startswith('network') else 'cycle{'
        prefix_strings = [step.to_str(format) for step in self.prefix_formulas]
        cycle_strings = [step.to_str(format) for step in self.cycle_formulas]
        if len(prefix_strings) > 0:
            prefix_strings.append('')  # for trailing ;
        return (';' + space).join(prefix_strings) + cycle_start + (';' + space).join(cycle_strings) + '}'

    def equal_to(self, other: 'LTLTrace', extended_eq=False):
        return len(self.prefix_formulas) == len(other.prefix_formulas) \
            and len(self.cycle_formulas) == len(other.cycle_formulas) \
            and all([my_step.equal_to(other_step, extended_eq=extended_eq) for my_step, other_step in zip(self.prefix_formulas, other.prefix_formulas)]) \
            and all([my_step.equal_to(other_step, extended_eq=extended_eq) for my_step, other_step in zip(self.cycle_formulas, other.cycle_formulas)])

    def contained_aps(self):
        return (reduce(operator.or_, [q.contained_aps() for q in self.prefix_formulas]) if self.prefix_formulas else set()) | reduce(operator.or_, [q.contained_aps() for q in self.cycle_formulas])

    def decide_disjunctions(self):
        assert self.symbolic
        self.prefix_formulas = [random.choice(dec_helper(step)) for step in self.prefix_formulas]
        self.cycle_formulas = [random.choice(dec_helper(step)) for step in self.cycle_formulas]


def dec_helper(node):
    if node.type_ == Token.OR:
        return dec_helper(node.lchild) + dec_helper(node.rchild)
    else:
        return [node]



class ParseError(Exception):
    pass


Token = Enum('Node', 'NOT AND OR IMPLIES EQUIV XOR NEXT UNTIL WEAKUNTIL RELEASE STRONGRELEASE GLOBALLY FINALLY TRUE FALSE AP LPAR RPAR STEP PAD EOS START UNK')

# token_dict_spot = {'!':(1, Token.NOT), '&':(2, Token.AND), '|':(2, Token.OR), '->':(2, Token.IMPLIES), '<->':(2, Token.EQUIV), 'xor':(2, Token.XOR), 'X':(1, Token.NEXT), 'U':(2, Token.UNTIL),
#         'W':(2, Token.WEAKUNTIL), 'R':(2, Token.RELEASE), 'G':(1, Token.GLOBALLY), 'F':(1, Token.FINALLY), '1':(0, Token.TRUE), '0':(0, Token.FALSE), '(':(-1, Token.LPAR), ')':(-1, Token.RPAR)}
# todo: handle "xor" appropriately
token_dict_spot = {'!':(1, Token.NOT), '&':(2, Token.AND), '|':(2, Token.OR), '->':(2, Token.IMPLIES), '<->':(2, Token.EQUIV), 'X':(1, Token.NEXT), 'U':(2, Token.UNTIL),
        'W':(2, Token.WEAKUNTIL), 'R':(2, Token.RELEASE), 'M':(2, Token.STRONGRELEASE), 'G':(1, Token.GLOBALLY), 'F':(1, Token.FINALLY), '1':(0, Token.TRUE), '0':(0, Token.FALSE), '(':(-1, Token.LPAR), ')':(-1, Token.RPAR)}
token_reverse_dict_spot = {token: ch for ch, (num_children, token) in token_dict_spot.items()}

token_dict_network = token_dict_spot.copy()
del token_dict_network['->']
token_dict_network['>'] = (2,Token.IMPLIES)
del token_dict_network['<->']
token_dict_network['='] = (2,Token.EQUIV)
token_reverse_dict_network = token_reverse_dict_spot.copy()
token_reverse_dict_network[Token.IMPLIES] = '>'
token_reverse_dict_network[Token.EQUIV] = '='

token_dict_lbt = {'!':(1, Token.NOT), '&':(2, Token.AND), '|':(2, Token.OR), 'i':(2, Token.IMPLIES), 'e':(2, Token.EQUIV), '^':(2, Token.XOR), 'X':(1, Token.NEXT), 'U':(2, Token.UNTIL),
        'W':(2, Token.WEAKUNTIL), 'R':(2, Token.RELEASE), 'M':(2, Token.STRONGRELEASE), 'G':(1, Token.GLOBALLY), 'F':(1, Token.FINALLY), 't':(0, Token.TRUE), 'f':(0, Token.FALSE)}
token_reverse_dict_lbt = {token: ch for ch, (num_children, token) in token_dict_lbt.items()}

precedence = {Token.NOT : 1, Token.AND : 3, Token.OR : 4, Token.IMPLIES : 5, Token.EQUIV : 6, Token.XOR : 6, Token.NEXT : 1, Token.UNTIL : 2, Token.WEAKUNTIL : 2, Token.RELEASE : 2, Token.STRONGRELEASE : 2, Token.GLOBALLY : 1, Token.FINALLY : 1, Token.TRUE : 0, Token.FALSE : 0, Token.AP : 0} # higher number = weaker
left_associative = {Token.AND : True, Token.OR: True, Token.IMPLIES : False, Token.EQUIV : None, Token.XOR : None, Token.UNTIL : False, Token.WEAKUNTIL : False, Token.RELEASE : False, Token.STRONGRELEASE : False}

sympy_tokens = {Token.NOT : syl.Not, Token.AND : syl.And, Token.OR : syl.Or, Token.IMPLIES : syl.Implies, Token.EQUIV : syl.Equivalent, Token.XOR : syl.Xor, Token.TRUE : syl.true, Token.FALSE : syl.false}

binary_ops = {Token.AND, Token.OR, Token.IMPLIES, Token.EQUIV, Token.UNTIL, Token.WEAKUNTIL} #, Token.RELEASE} # todo: clean handling of release
unary_ops = {Token.NOT, Token.NEXT, Token.GLOBALLY, Token.FINALLY}
boolean_tokens = {Token.NOT, Token.AND, Token.OR, Token.IMPLIES, Token.EQUIV, Token.TRUE, Token.FALSE, Token.AP}
ltl_tokens = {Token.NEXT, Token.UNTIL, Token.WEAKUNTIL, Token.RELEASE, Token.GLOBALLY, Token.FINALLY}


class LTLFormulaBinaryOp(LTLFormula):
    def __init__(self, type_, lchild, rchild):
        self.type_ = type_
        self.lchild = lchild
        self.rchild = rchild
        self.precedence = precedence[type_]
        self.left_associative = left_associative[type_]

    def _to_str(self, notation, format_, spacing, full_parens):
        space = '' if spacing == 'none' else ' '
        if notation == 'polish':
            return globals()['token_reverse_dict_' + format_][self.type_] + space + self.lchild._to_str(notation=notation, format_=format_, spacing=spacing, full_parens=full_parens) + space + self.rchild._to_str(notation=notation, format_=format_, spacing=spacing, full_parens=full_parens)
        elif notation == 'infix':
            if full_parens or self.lchild.precedence > self.precedence:
                par_left = True
            elif self.lchild.precedence == self.precedence:
                par_left = self.left_associative is None or not self.left_associative
            else:
                par_left = False
            if full_parens or self.rchild.precedence > self.precedence:
                par_right = True
            elif self.rchild.precedence == self.precedence:
                par_right = self.left_associative is None or self.left_associative
            else:
                par_right = False
            return ('(' if par_left else '') + self.lchild._to_str(notation=notation, format_=format_, spacing=spacing, full_parens=full_parens) + (')' if par_left else '') + space + globals()['token_reverse_dict_' + format_][self.type_] + space + ('(' if par_right else '') + self.rchild._to_str(notation=notation, format_=format_, spacing=spacing, full_parens=full_parens) + (')' if par_right else '')
        else:
            raise ValueError("Unrecognized notation")

    def equal_to(self, other: LTLFormula, extended_eq=False):
        if not isinstance(other, LTLFormulaBinaryOp) or not self.type_ == other.type_:
            return False
        children_equal = self.lchild.equal_to(other.lchild, extended_eq=extended_eq) and self.rchild.equal_to(other.rchild, extended_eq=extended_eq)
        if extended_eq and self.type_ in [Token.AND, Token.OR]:
            children_equal = children_equal or (self.lchild.equal_to(other.rchild, extended_eq=extended_eq) and self.rchild.equal_to(other.lchild, extended_eq=extended_eq))
        return children_equal

    def size(self):
        return 1 + self.lchild.size() + self.rchild.size()

    def contained_aps(self):
        return self.lchild.contained_aps() | self.rchild.contained_aps()

    def rewrite(self, token):
        lchild_r = self.lchild.rewrite(token)
        rchild_r = self.rchild.rewrite(token)
        if self.type_ == token:
            if token == Token.OR:
                return F_NOT(F_AND(F_NOT(lchild_r), F_NOT(rchild_r)))
            elif token == Token.WEAKUNTIL:
                return F_RELEASE(rchild_r, F_OR(lchild_r, rchild_r))
            else:
                raise ValueError("Don't know how to rewrite " + str(token))
        else:
            return LTLFormulaBinaryOp(self.type_, lchild_r, rchild_r)

    def binary_position_list(self, format, add_first):
        lchild_pos_list = self.lchild.binary_position_list(format, add_first)
        rchild_pos_list = self.rchild.binary_position_list(format, add_first)
        if add_first:
            # due to the recursion the branching choice is added at the end of the list if add_first is true
            lchild_pos_list = [l + [1, 0] for l in lchild_pos_list]
            rchild_pos_list = [l + [0, 1] for l in rchild_pos_list]
        else:
            lchild_pos_list = [[1, 0] + l for l in lchild_pos_list]
            rchild_pos_list = [[0, 1] + l for l in rchild_pos_list]
        if format == 'lbt':
            return [[]] + lchild_pos_list + rchild_pos_list
        elif format == 'spot':
            return lchild_pos_list + [[]] + rchild_pos_list
        else:
            raise ValueError("Unrecognized format")

    def to_sympy(self):
        return sympy_tokens[self.type_](self.lchild.to_sympy(), self.rchild.to_sympy())

    def __iter__(self):
        return IterHelper(self)
    
    def negation_normal_form(self, negate=False):
        if negate:
            if self.type_ == Token.AND:
                return F_OR(self.lchild.negation_normal_form(True), self.rchild.negation_normal_form(True))
            if self.type_ == Token.OR:
                return F_AND(self.lchild.negation_normal_form(True), self.rchild.negation_normal_form(True))
            if self.type_ == Token.IMPLIES:
                return F_AND(self.lchild.negation_normal_form(False), self.rchild.negation_normal_form(True))
            if self.type_ == Token.EQUIV:
                return F_AND(F_OR(self.lchild.negation_normal_form(False), self.rchild.negation_normal_form(False)),
                    F_OR(self.lchild.negation_normal_form(True), self.rchild.negation_normal_form(True)))
            if self.type_ == Token.XOR:
                return F_OR(F_AND(self.lchild.negation_normal_form(False), self.rchild.negation_normal_form(False)),
                    F_AND(self.lchild.negation_normal_form(True), self.rchild.negation_normal_form(True)))
            if self.type_ == Token.UNTIL:
                return LTLFormulaBinaryOp(Token.RELEASE, self.lchild.negation_normal_form(True), self.rchild.negation_normal_form(True))
            if self.type_ == Token.WEAKUNTIL:
                return LTLFormulaBinaryOp(Token.UNTIL, self.rchild.negation_normal_form(True), 
                    F_AND(self.lchild.negation_normal_form(True), self.rchild.negation_normal_form(True)))
            if self.type_ == Token.RELEASE:
                return LTLFormulaBinaryOp(Token.UNTIL, self.lchild.negation_normal_form(True), self.rchild.negation_normal_form(True))
            raise NotImplementedError()
        else:
            if self.type_ == Token.IMPLIES:
                return F_OR(self.lchild.negation_normal_form(True), self.rchild.negation_normal_form(False))
            if self.type_ == Token.EQUIV:
                return F_OR(F_AND(self.lchild.negation_normal_form(False), self.rchild.negation_normal_form(False)),
                        F_AND(self.lchild.negation_normal_form(True), self.rchild.negation_normal_form(True)))
            if self.type_ == Token.XOR:
                return F_AND(F_OR(self.lchild.negation_normal_form(False), self.rchild.negation_normal_form(False)),
                    F_OR(self.lchild.negation_normal_form(True), self.rchild.negation_normal_form(True)))
            if self.type_ == Token.WEAKUNTIL:
                return LTLFormulaBinaryOp(Token.RELEASE, self.rchild.negation_normal_form(False),
                        F_OR(self.lchild.negation_normal_form(False), self.rchild.negation_normal_form(False)))
            if self.type_ in [Token.AND, Token.OR, Token.UNTIL, Token.RELEASE]:
                return LTLFormulaBinaryOp(self.type_, self.lchild.negation_normal_form(False), self.rchild.negation_normal_form(False))
            raise NotImplementedError()

    def relax_to_prop(self):
        lchild, rchild, = self.lchild.relax_to_prop(), self.rchild.relax_to_prop()
        if self.type_ == Token.UNTIL:
            return F_OR(lchild, rchild)
        if self.type_ == Token.RELEASE:
            return F_OR(rchild, F_AND(lchild, rchild))
        if self.type_ in [Token.AND, Token.OR]:
            return LTLFormulaBinaryOp(self.type_, lchild, rchild)
        raise ValueError('Require negation normal form')



class LTLFormulaUnaryOp(LTLFormula):
    def __init__(self, type_, child):
        self.type_ = type_
        self.child = child
        self.precedence = precedence[type_]

    def _to_str(self, notation, format_, spacing, full_parens):
        space = '' if spacing in ['none', 'binary ops'] else ' '
        if notation == 'polish':
            return globals()['token_reverse_dict_' + format_][self.type_] + space + self.child._to_str(notation=notation, format_=format_, spacing=spacing, full_parens=full_parens)
        elif notation == 'infix':
            par = (self.child.precedence > self.precedence) or full_parens
            return globals()['token_reverse_dict_' + format_][self.type_] + space + ('(' if par else '') + self.child._to_str(notation=notation, format_=format_, spacing=spacing, full_parens=full_parens) + (')' if par else '')
        else:
            raise ValueError("Unrecognized notation")

    def equal_to(self, other: LTLFormula, extended_eq=False):
        return isinstance(other, LTLFormulaUnaryOp) and self.type_ == other.type_ and self.child.equal_to(other.child, extended_eq=extended_eq)

    def size(self):
        return 1 + self.child.size()

    def contained_aps(self):
        return self.child.contained_aps()

    def rewrite(self, token):
        child_r = self.child.rewrite(token)
        if self.type_ == token:
            if token == Token.GLOBALLY:
                raise NotImplementedError()
            else:
                raise ValueError("Don't know how to rewrite " + str(token))
        else:
            return LTLFormulaUnaryOp(self.type_, child_r)

    def binary_position_list(self, format, add_first):
        child_pos_list = self.child.binary_position_list(format, add_first)
        if add_first:
            child_pos_list = [l + [1, 0] for l in child_pos_list]
        else:
            child_pos_list = [[1, 0] + l for l in child_pos_list]
        if format == 'lbt' or format == 'spot':
            return [[]] + child_pos_list
        else:
            raise ValueError("Unrecognized format")

    def to_sympy(self):
        return sympy_tokens[self.type_](self.child.to_sympy())

    def __iter__(self):
        return IterHelper(self)


    def negation_normal_form(self, negate=False):
        if negate:
            if self.type_ == Token.NOT:
                return self.child.negation_normal_form(False)
            if self.type_ == Token.NEXT:
                return F_NEXT(self.child.negation_normal_form(True))
            if self.type_ == Token.FINALLY:
                return LTLFormulaBinaryOp(Token.RELEASE, LTLFormulaLeaf(Token.FALSE), self.child.negation_normal_form(True))
            if self.type_ == Token.GLOBALLY:
                return LTLFormulaBinaryOp(Token.UNTIL, LTLFormulaLeaf(Token.TRUE), self.child.negation_normal_form(True))
        else:
            if self.type_ == Token.NOT:
                return self.child.negation_normal_form(True)
            if self.type_ == Token.NEXT:
                return F_NEXT(self.child.negation_normal_form(False))
            if self.type_ == Token.FINALLY:
                return LTLFormulaBinaryOp(Token.UNTIL, LTLFormulaLeaf(Token.TRUE), self.child.negation_normal_form(False))
            if self.type_ == Token.GLOBALLY:
                return LTLFormulaBinaryOp(Token.RELEASE, LTLFormulaLeaf(Token.FALSE), self.child.negation_normal_form(False))
            raise NotImplementedError()


    def relax_to_prop(self):
        if self.type_ == Token.NEXT:
            return LTLFormulaLeaf(Token.TRUE)
        if self.type_ == Token.NOT:
            if not isinstance(self.child, LTLFormulaLeaf):
                raise ValueError('Require negation normal form!')
            return LTLFormulaUnaryOp(Token.NOT, self.child.relax_to_prop())
        raise NotImplementedError()


class LTLFormulaLeaf(LTLFormula):
    def __init__(self, type_, ap=None):
        self.type_ = type_
        self.ap = ap
        self.precedence = precedence[type_]

    def _to_str(self, notation, format_, spacing, full_parens):
        if not self.type_ == Token.AP:
            return globals()['token_reverse_dict_' + format_][self.type_]
        if format_ == 'lbt':
            return '"' + self.ap + '"'
        elif format_ == 'spot' or format_ == 'network':
            return self.ap
        else:
            raise ValueError("'format' must be either spot or lbt")

    def equal_to(self, other: LTLFormula, extended_eq=False):
        return isinstance(other, LTLFormulaLeaf) and self.type_ == other.type_ and self.ap == other.ap

    def size(self):
        return 1

    def contained_aps(self):
        if self.type_ == Token.AP:
            return {self.ap}
        else:
            return set()

    def rewrite(self, token):
        if self.type_ == token:
            raise ValueError("Cannot rewrite a " + str(token))
        else:
            return LTLFormulaLeaf(self.type_, ap=self.ap)

    def binary_position_list(self, format, add_first):
        return [[]]

    def to_sympy(self):
        if self.type_ == Token.AP:
            return sy.symbols(self.ap)
        else:
            return sympy_tokens[self.type_]

    def __iter__(self):
        return IterHelper(self)

    def negation_normal_form(self, negate=False):
        if negate:
            if self.type_ == Token.TRUE:
                return LTLFormulaLeaf(Token.FALSE)
            if self.type_ == Token.FALSE:
                return LTLFormulaLeaf(Token.TRUE)
            if self.type_ == Token.AP:
                return LTLFormulaUnaryOp(Token.NOT, LTLFormulaLeaf(Token.AP, ap=self.ap))
        else:
            return LTLFormulaLeaf(self.type_, ap=self.ap)
    
    def relax_to_prop(self):
        return LTLFormulaLeaf(self.type_, ap=self.ap)
        


class IterHelper:
    def __init__(self, target):
        self.stack = [(target, 0)]

    def __next__(self):
        if len(self.stack) == 0:
            raise StopIteration

        target, step = self.stack.pop()
        if step == 0:
            if isinstance(target, LTLFormulaBinaryOp):
                self.stack.append((target, 1))
            elif isinstance(target, LTLFormulaUnaryOp):
                self.stack.append((target, 1))
            elif isinstance(target, LTLFormulaLeaf):
                pass
            else:
                raise ValueError
            return target
        elif step == 1:
            if isinstance(target, LTLFormulaBinaryOp):
                self.stack.append((target, 2))
                self.stack.append((target.lchild, 0))
            elif isinstance(target, LTLFormulaUnaryOp):
                self.stack.append((target.child, 0))
            else:
                raise ValueError
            return self.__next__()
        elif step == 2:
            assert isinstance(target, LTLFormulaBinaryOp)
            self.stack.append((target.rchild, 0))
            return self.__next__()
        else:
            raise ValueError


def tokenize_formula(formula_string, format_, return_statistics=False):
    token_dict = globals()['token_dict_' + format_]
    token_list = []
    stats = {}
    while formula_string:
        ap_p_numeric_match = re.match(ap_p_numeric_re, formula_string)
        if ap_p_numeric_match:
            name = ap_p_numeric_match.group()
            token_list.append((0, Token.AP, name))
            stats[name] = stats.get(name, 0) + 1
            formula_string = formula_string[ap_p_numeric_match.end():]
            continue
        if len(formula_string) >= 5:
            if formula_string[:5] == '<pad>':
                token_list.append((-1, Token.PAD))
                stats['<pad>'] = stats.get('<pad>', 0) +1
                formula_string = formula_string[5:]
                continue
            elif formula_string[:5] == '<eos>':
                token_list.append((-1, Token.EOS))
                stats['<eos>'] = stats.get('<eos>', 0) +1
                formula_string = formula_string[5:]
                continue
        if len(formula_string) >= 7:
            if formula_string[:7] == '<start>':
                token_list.append((-1, Token.START))
                stats['<start>'] = stats.get('<start>', 0) +1
                formula_string = formula_string[7:]
                continue
        if len(formula_string) >= 3:
            token = token_dict.get(formula_string[:3]) # 3 character match (ugly, damn)
            if token:
                token_list.append(token)
                stats[formula_string[:3]] = stats.get(formula_string[:3], 0) + 1
                formula_string = formula_string[3:]
                continue
        if len(formula_string) >= 2:
            token = token_dict.get(formula_string[:2]) # 2 character match (ugly, damn)
            if token:
                token_list.append(token)
                stats[formula_string[:2]] = stats.get(formula_string[:2], 0) + 1
                formula_string = formula_string[2:]
                continue
        c = formula_string[:1]
        formula_string = formula_string[1:]
        if c.isspace() and format_ != 'network':
            continue
        token = token_dict.get(c)
        if token:
            token_list.append(token)
            stats[c] = stats.get(c, 0) + 1
        elif (format_ == 'spot' or format_ == 'network') and re.match(ap_alphabetic_re, c):  # check for AP a
            token_list.append((0, Token.AP, c))
            stats[c] = stats.get(c, 0) + 1
        elif format_ == 'lbt' and len(formula_string) >= 2 and re.match('"' + ap_alphabetic_re + '"', c + formula_string[0] + formula_string[1]):  # check for AP "a"
            token_list.append((0, Token.AP, formula_string[0]))
            stats[formula_string[0]] = stats.get(formula_string[0], 0) + 1
            formula_string = formula_string[2:]
        else:
            raise ParseError("Cannot tokenize '" + c + "', remainder '" + formula_string + "'")
    if return_statistics:
        return token_list, stats
    else:
        return token_list


def parse_polish_formula(token_list):
    if len(token_list) == 0:
        raise ParseError('Attempt to parse from empty token list')
    num_children, type_, *name = token_list.pop(0)
    if num_children == 2:
        lchild, token_list = parse_polish_formula(token_list)
        rchild, token_list = parse_polish_formula(token_list)
        return LTLFormulaBinaryOp(type_, lchild, rchild), token_list
    elif num_children == 1:
        child, token_list = parse_polish_formula(token_list)
        return LTLFormulaUnaryOp(type_, child), token_list
    elif num_children == 0:
        if type_ == Token.AP:
            return LTLFormulaLeaf(type_, ap=name[0]), token_list
        else:
            return LTLFormulaLeaf(type_, ap=None), token_list
    else:
        raise ParseError("Illegal token '" + str(type_) + "'")


def parse_infix_formula(token_list, expect_rpar=False):
    # first part, until possible first binary op
    node, token_list = infix_parse_single(token_list)
    if len(token_list) == 0:
        if expect_rpar:
            raise ParseError("Parsing error: End of string but expected RPAR")
        else:
            return node, []
    num_children, type_, *_ = token_list.pop(0)
    if expect_rpar and type_ == Token.RPAR:
        return node, token_list
    if num_children != 2:
        raise ParseError("Parsing error: Binary operator expected, got " + str(type_) + ", remainder: " + str(token_list))

    # main part, at least one binary op
    stack = [(node, type_)]
    while True:
        current_node, token_list = infix_parse_single(token_list)
        l = len(token_list)
        if l == 0 and expect_rpar:
            raise ParseError("Parsing error: End of string but expected RPAR")
        if l > 0:
            num_children, right_op, *_ = token_list.pop(0)
        if l == 0 or (expect_rpar and right_op == Token.RPAR):  # finished
            assert len(stack) > 0
            while len(stack) > 0:
                left_node, left_op = stack.pop()
                current_node = LTLFormulaBinaryOp(left_op, left_node, current_node)
            return current_node, token_list

        # not yet finished, binary op present
        if num_children != 2:
            raise ParseError("Parsing error: Binary operator expected, got " + str(right_op) + ", remainder: " + str(token_list))
        left_node, left_op = stack[-1]
        while precedence[left_op] < precedence[right_op] or (precedence[left_op] == precedence[right_op] and left_associative[left_op]):  # left is stronger, apply left
            current_node = LTLFormulaBinaryOp(left_op, left_node, current_node)
            stack.pop()
            if len(stack) == 0:
                break
            left_node, left_op = stack[-1]
        stack.append((current_node, right_op))


def infix_parse_single(token_list):
    if len(token_list) == 0:
        raise ParseError('Attempt to parse from empty token list (trailing part missing?)')
    num_children, type_, *name = token_list.pop(0)
    if num_children == 2:
        raise ParseError("Parsing error: Binary operator at front (" + str(type_) + "), remainder: " + str(token_list))
    elif num_children == 1:
        child, token_list = infix_parse_single(token_list)
        return LTLFormulaUnaryOp(type_, child), token_list
    elif num_children == 0:
        if type_ == Token.AP:
            return LTLFormulaLeaf(type_, ap=name[0]), token_list
        else:
            return LTLFormulaLeaf(type_, ap=None), token_list
    else:
        if type_ == Token.RPAR:
            raise ParseError("Parsing error: RPAR at front, remainder: " + str(token_list))
        if type_ == Token.LPAR:
            return parse_infix_formula(token_list, expect_rpar=True)


sympy_token_dict = { syl.And : (2, Token.AND), syl.Or : (2, Token.OR), syl.Implies : (2, Token.IMPLIES), syl.Equivalent : (2, Token.EQUIV), syl.Xor : (2, Token.XOR), syl.Not : (1, Token.NOT), sy.core.symbol.Symbol : (0, Token.AP)}

def from_sympy(formula_sympy):
    t = type(formula_sympy)
    children, token = sympy_token_dict.get(t, (0, None))
    if children == 2:
        args = list(formula_sympy.args)
        current = LTLFormulaBinaryOp(token, from_sympy(args.pop(0)), from_sympy(args.pop(0)))
        while len(args) > 0:
            current = LTLFormulaBinaryOp(token, current, from_sympy(args.pop(0)))
        return current
    elif children == 1:
        return LTLFormulaUnaryOp(token, from_sympy(*formula_sympy.args))
    elif token == Token.AP:
        return LTLFormulaLeaf(token, ap=formula_sympy.name)
    elif formula_sympy == syl.true:
        return LTLFormulaLeaf(Token.TRUE)
    elif formula_sympy == syl.false:
        return LTLFormulaLeaf(Token.FALSE)
    else:
        raise ValueError()


def F_AND(x, y):
    return LTLFormulaBinaryOp(Token.AND, x, y)

def F_OR(x, y):
    return LTLFormulaBinaryOp(Token.OR, x, y)

def F_IMPLIES(x, y):
    return LTLFormulaBinaryOp(Token.IMPLIES, x, y)


def F_NEXT(x):
    return LTLFormulaUnaryOp(Token.NEXT, x)


def F_GLOBALLY(x):
    return LTLFormulaUnaryOp(Token.GLOBALLY, x)


def F_NOT(x):
    return LTLFormulaUnaryOp(Token.NOT, x)


def F_AP(s):
    return LTLFormulaLeaf(Token.AP, s)

def F_RELEASE(x, y):
    return LTLFormulaBinaryOp(Token.RELEASE, x, y)