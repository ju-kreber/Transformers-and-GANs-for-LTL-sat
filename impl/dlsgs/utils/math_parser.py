import re

from dlsgs.utils.ltl_parser import ParseError

token_dict = {'CONST':None, '<pad>':None, '<eos>':None, 'x':0, 'y':0, 'z':0, 'E':0, 'pi':0, 'INT+':None, 'INT-':None, 'add':2, 'mul':2, 'div':2, 'pow':2, 'exp':1, 'sin':1, 'cos':1, 'tan':1, 'asin':1, 'acos':1, 'atan':1, 'sinh':1,
     'cosh':1, 'tanh':1, 'asinh':1, 'acosh':1, 'atanh':1, 'sqrt':1, 'ln':1, 'abs':1, 'sign':1, 'sec':1, 'csc':1, 'cot':1, 'asec':1, 'acot':1, 'coth':1, 'asech':1, 'acoth':1, 'acsch':1,
     'f':1, 'g':1, 'h':1, 'pow2':1, 'pow3':1, 'pow4':1, 'pow5':1}

class Node():
    def __init__(self, token, children=None, val=None):
        self.token = token
        self.val = val
        self.children = children if children is not None else []

    def to_str(self, format):
        assert format == 'network-polish'
        s = self.token
        if self.token in ['INT+', 'INT-']:
            s += ' ' + ' '.join(self.val)
        return ' '.join([s] + [q.to_str(format) for q in self.children])
    
    def as_list(self):
        l = [self.token]
        if self.token in ['INT+', 'INT-']:
            l.extend(self.val)
        for q in self.children:
            l.extend(q.as_list())
        return l
    
    def pretty(self):
        if self.val is not None:
            val = ''.join(self.val)
            if self.token == 'INT-':
                val = '-' + val
            else:
                assert self.token == 'INT+'
            assert len(self.children) == 0
            return val
        if len(self.children) == 2:
            return '(' + self.children[0].pretty() + ') ' + self.token + ' (' + self.children[1].pretty() + ')'
        elif len(self.children) == 1:
            return self.token + ' ' + self.children[0].pretty()
        elif len(self.children) == 0:
            return self.token
        else:
            raise ValueError('Illegal number of children')
        


def parse(line):
    m = re.fullmatch(r"^\d+\|sub Y' (.*)\t.*$", line)
    if not m:
        raise ParseError('Could not match ' + line)
    tokens = m.groups()[0].split(' ')
    if len(tokens) == 0:
        raise ParseError('Empty token list')
    formula, remainder = parse_help(tokens)
    if len(remainder) > 0:
        raise ParseError('original line: ' + line + '\nformula: ' + formula.to_str('network-polish') + '\nremainder' + ' '.join(remainder))
    return formula


def parse_help(tokens):
    if len(tokens) == 0:
        raise ParseError('try to parse empty list!')
    token, tokens = tokens[0], tokens[1:]
    if token in ['INT+', 'INT-']:
        numbers = []
        while len(tokens) > 0 and re.match(r'\d', tokens[0]):
            numbers.append(tokens.pop(0))
        if len(numbers) == 0:
            raise ParseError('INT not followed by number!')
        return Node(token, val=numbers), tokens
    arity = token_dict.get(token, None)
    if arity is None:
        raise ParseError('Illegal token: ' + token)
    children = []
    for i in range(arity):
        node, tokens = parse_help(tokens)
        children.append(node)
    return Node(token, children=children), tokens
        
