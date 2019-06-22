"""Taxonomy (tree) helper functions

"""
from anytree import Node, RenderTree
import logging

logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')

def parse_tree(s, h_sep=' ', w_sep='\n'):
    ''' parse tree in simple text format '''
    s = [(x.count(h_sep), x.replace(h_sep, '').strip()) for x in s.split(w_sep)]
    s = [x for x in s if len(x[1])>0]
    
    root = Node('root', level=-1)
    nodes = [root] + [Node(name, level=i) for i, name in s]
    
    for i in range(len(nodes)):
        for j in range(i-1, -1, -1):
            if nodes[j].level < nodes[i].level:
                nodes[i].parent = nodes[j]
                break
                
    return root
    
def generate_sample_taxonomy():
    
    simple_tax = Node('root')
    
    A = Node('A', parent=simple_tax)
    B = Node('B', parent=simple_tax)
    C = Node('C', parent=simple_tax)
    
    A1 = Node('A1', parent=A)
    A2 = Node('A2', parent=A)
    A3 = Node('A3', parent=A)
    A4 = Node('A4', parent=A)
    
    B1 = Node('B1', parent=B)
    B2 = Node('B2', parent=B)
    B3 = Node('B3', parent=B)
    
    C1 = Node('C1', parent=C)
    C2 = Node('C2', parent=C)
    C3 = Node('C3', parent=C)
    C4 = Node('C4', parent=C)  
    
    return simple_tax


def print_tree(tax, only_name=True, attrs=None):
    if only_name:
        for pre, _, node in RenderTree(tax):
            print(f'{pre}{node.name}')
    elif attrs is not None:
        for pre, _, node in RenderTree(tax):
            print(f'{pre}{node.name}: ', end='')
            print(', '.join([f'{x}={getattr(node, x)}' for x in attrs]))
    else:
        print(RenderTree(tax))
    