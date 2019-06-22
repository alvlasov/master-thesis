"""Parsimonious Generalization of Fuzzy Sets

"""
import sys
from copy import deepcopy
from anytree import Node, LevelOrderIter, RenderTree, findall, NodeMixin
from terminaltables import AsciiTable
from termcolor import colored

import logging
import numpy as np
logger = logging.getLogger(__name__)
logger.setLevel('INFO')

from taxonomy import generate_sample_taxonomy

EPS = 1e-16


class PGFS:
    """Parsimonious Generalization of Fuzzy Sets
    
    This implementation follows the article by Mirkin, Fenner, Nascimento and 
    Felizardo (2016)
    
    Parameters
    ---------
    taxonomy: Node
        Root of the taxonomy tree.
    
    lmbda : float
        Gap penalty.
        
    gamma : float 
        Offshoot penalty.
       
    
    """    
    
    def __init__(self, taxonomy, lmbda, gamma):
        
        assert isinstance(taxonomy, NodeMixin)
        assert isinstance(lmbda, (float, int))
        assert isinstance(gamma, (float, int))
        
        self.taxonomy = taxonomy
        self._init_event_counts()
        
        self.lmbda = lmbda
        self.gamma = gamma
        
    def _init_event_counts(self):

        self.taxonomy.n_samples = 0

        for n in LevelOrderIter(self.taxonomy):
            n.n_gains = 0
            n.n_losses = 0
    
    def _assign_membership(self, u_dict):
        
        tax = self.query_lifted
        
        for name, u in u_dict.items():
            t = findall(tax, filter_=lambda x: x.name == name and x.is_leaf)
            assert len(t) > 0, f'there is no leaf "{name}" in a taxonomy!'
            assert len(t) == 1, f'leaf "{name}" is non-unique!'
            t[0].u = u

        leaves = [t for t in LevelOrderIter(tax, 
                                            filter_=lambda t: t.is_leaf)]
        
        for leaf in leaves:
            if not hasattr(leaf, 'u'):
                leaf.u = 0

    def _normalize_membership(self):
        ''' sum of u(t)^2 = 1'''
        
        tax = self.query_lifted
        # iterate over leaves
        # collect all leaves to array
        leaves = [t for t in LevelOrderIter(tax, 
                                            filter_=lambda t: t.is_leaf)]
        logger.debug(f'leaves: {leaves}')
        # collect all non-zero u-values
        u_values = [t.u for t in leaves if hasattr(t, 'u') and t.u > 0]
        logger.debug(f'non-zero membership values: {u_values}')
        
        u_values = np.asarray(u_values)
        assert len(u_values) > 0, 'there are no non-zero membership values!'
        assert np.all(u_values >= 0), 'there are negative membership values!'
    
        # compute normalization coefficient
        c = np.sqrt(np.sum(u_values ** 2))
        logger.debug(f'normalization coeff: {c}')
        
        # normalize
        for t in leaves:
            if not hasattr(t, 'u'):
                t.u = 0
            else:
                t.u /= c
            
        u_values = [t.u for t in leaves]
        logger.debug(f'new membership values: {u_values}')
        
    def _lift_membership(self):
        
        tax = self.query_lifted
        
        # iterate level-order starting at leaves-1 level
        branches = [t for t in LevelOrderIter(tax, filter_=lambda t: not t.is_leaf)]
        
        logger.debug(f'branches: {branches}')
        # from bottom to top
        
        # fill memberships s.t. sum of u**2 of children = 1
        for t in reversed(branches):
            logger.debug(f'{t} children: {t.children}')
            u = [c.u for c in t.children]
            u = np.asarray(u)
            t.u = np.sqrt(np.sum(u ** 2))
    
    def _prune(self):
    
        tax = self.query_lifted
        
        for t in LevelOrderIter(tax):
            assert hasattr(t, 'u'), f'membership value for node {t.parent.name} is missing!'
            if t.u == 0:
                for c in t.children:
                    c.parent = None
                    
    def _compute_gap_statistics(self):
        
        tax = self.query_lifted
        
        nodes = list(LevelOrderIter(tax))
        
        for t in reversed(nodes):
            
            # fill gap importances
            if not t.is_root:
                assert hasattr(t.parent, 'u'), f'membership value for node {t.parent.name} is missing!'
                t.v = t.parent.u if t.u == 0 else 0
                
            logger.debug(f'fill_gap: {t} : {t.children}')
            
            # fill gap summary importances
            t.V = t.v if t.is_leaf else sum([c.V for c in t.children]) 
            
            # fill gap sets
            if t.is_leaf:
                t.G = {t.name} if t.u == 0 else set()
            else:
                t.G = set.union(*[c.G for c in t.children]) 
                
    def _lift(self):
        
        lmbda = self.lmbda
        gamma = self.gamma
        tax = self.query_lifted
        
        nodes = list(LevelOrderIter(tax))
        
        for t in reversed(nodes):
            if t.is_leaf:
                # gain, if u>0 else nothing
                t.H = {t.name} if t.u > 0 else set()
                t.L = set() if t.u > 0 else set()
                t.p = gamma*t.u if t.u > 0 else 0
            else:
                p_children = sum([c.p for c in t.children])
                if t.u + lmbda*t.V <= p_children:
                    t.H = {t.name}
                    t.L = t.G
                    t.p = t.u + lmbda*t.V
                else:
                    t.H = set.union(*(c.H for c in t.children))
                    t.L = set.union(*(c.L for c in t.children))
                    t.p = sum((c.p for c in t.children))
                    
    def _update_event_counts(self):

        root = self.query_lifted
        self.taxonomy.n_samples += 1

        for n in root.H:
            t = findall(self.taxonomy, filter_=lambda t: t.name == n)
            t[0].n_gains += 1

        for n in root.L:
            t = findall(self.taxonomy, filter_=lambda t: t.name == n)
            t[0].n_losses += 1

    def lift(self, membership_dict, output=False):
        
        self.query_lifted = deepcopy(self.taxonomy)
        
        self._assign_membership(membership_dict)
        self._normalize_membership()
        self._lift_membership()
        self._prune()
        self._compute_gap_statistics()
        self._lift()
        self._update_event_counts()

        if output is True:
            self.print_results()
        elif output is not False:
            self.print_results(output)

        return self.query_lifted

    def estimate_ml(self):

        tax = self.taxonomy
        n_samples = tax.n_samples

        assert n_samples > 1, 'you must lift at least 2 samples to estimate probabilities'

        for n in LevelOrderIter(tax):
            n.p_gain = n.n_gains / n_samples
            n.p_loss = n.n_losses / n_samples

        return tax
    
    def lift_ml(self, membership_dict, output=False):

        self.query_lifted = deepcopy(self.taxonomy)
        
        self._assign_membership(membership_dict)
        # self._normalize_membership()
        self._lift_membership()
        self._prune()

        tax = self.query_lifted

        nodes = list(LevelOrderIter(tax))
        
        for t in reversed(nodes):
            if t.is_leaf:
                t.log_p_inherited = np.log((1-2*t.p_loss) * t.u + t.p_loss + EPS)
                t.log_p_not_inherited = np.log((2*t.p_gain - 1) * t.u + 1 - t.p_gain + EPS)
                t.lost = ''
                t.gain = ''
                t.p_inherited_all = (0, 0)
                t.p_not_inherited_all = (0,0)
                t.events_i = []
                if t.u == 0:
                    t.events_i += [(t.name, 'loss')]
                t.events_n = []
                if t.u > 0:
                    t.events_n += [(t.name, 'gain')]
                logger.debug(f'ml: leaf "{t.name}" u = {t.u:.3f}')
                logger.debug(f'ml: leaf "{t.name}" p_g = {t.p_gain:.3f}')
                logger.debug(f'ml: leaf "{t.name}" p_l = {t.p_loss:.3f}')
                logger.debug(f'ml: leaf "{t.name}" p_i = {np.exp(t.log_p_inherited):.3f}')
                logger.debug(f'ml: leaf "{t.name}" p_n = {np.exp(t.log_p_not_inherited):.3f}')
            else:
                children = t.children

                sum_log_p_inherited = sum([c.log_p_inherited for c in children])
                sum_log_p_not_inherited = sum([c.log_p_not_inherited for c in children])

                scenario_lost = np.log(t.p_loss + EPS) + sum_log_p_not_inherited
                scenario_not_lost = np.log(1-t.p_loss + EPS) + sum_log_p_inherited

                t.log_p_inherited = max(scenario_lost, scenario_not_lost)
                t.p_inherited_all = np.exp(scenario_lost), np.exp(scenario_not_lost)

                if scenario_lost > scenario_not_lost: # I->L
                    t.events_i = sum([c.events_n for c in children], []) + [(t.name, 'loss')]
                    # while len(t.events_i) > 0 and t.events_i[-1][1] == 'loss':
                    #     t.events_i.pop()
                    # t.events_i += [(t.name, 'loss')]
                else:  # I-> not L
                    t.events_i = sum([c.events_i for c in children], [])

                scenario_gain = np.log(t.p_gain + EPS) + sum_log_p_inherited
                scenario_no_gain = np.log(1-t.p_gain + EPS) + sum_log_p_not_inherited

                t.log_p_not_inherited = max(scenario_gain, scenario_no_gain)
                t.p_not_inherited_all = np.exp(scenario_gain), np.exp(scenario_no_gain)
                # t.events_n = sum([c.events_n for c in children], [])
                if scenario_gain > scenario_no_gain: # N -> G
                    t.events_n = sum([c.events_i for c in children], []) + [(t.name, 'gain')]
                    # while len(t.events_n) > 0 and t.events_n[-1][1] == 'gain':
                    #     t.events_n.pop()
                    # t.events_n += [(t.name, 'gain')]
                else:
                    t.events_n = sum([c.events_n for c in children], [])

            t.p_inherited = np.exp(t.log_p_inherited)
            t.p_not_inherited = np.exp(t.log_p_not_inherited)

        return tax

    def print_results(self, output=None):
        ''' 
        print results to output stream 
        if output is None, use stdout
        '''

        assert hasattr(self, 'query_lifted'), 'you should apply lift_fuzzy_set() to some fuzzy set'
        
        table = []
        table.append(['tree', 'u', 'H', 'L', 'p'])
        for pre, _, t in RenderTree(self.query_lifted):
            h_str = str(t.H) if len(t.H)>0 else ""
            l_str = str(t.L) if len(t.L)>0 else ""
            p_str = f'{t.p:.4f}' if t.p > 0 else '0'
            u = f'{t.u:.3f}' if t.is_leaf and t.u > 0 else ''
            s = pre + (t.name if u == '' or output is not None else colored(t.name, 'green'))
            table.append([s, u, h_str, l_str, p_str])
            
        if output is None:
            output = sys.stdout

        print(AsciiTable(table).table, file=output)


def test_simple_taxonomy():
    ''' 
    test on the simple taxonomy given in Mirkin, Fenner, Nascimento, 
    Felizardo (2016) 
    '''
    
    simple_tax = generate_sample_taxonomy()
    
    data = [
        (({'A1': 0.8, 'A2': 0.5, 'B1': 0.1, 'B2': 0.01}, 0.9, 0.2), 
         ({'A1', 'A2', 'B1', 'B2'}, set(), 1.34)),
         
        (({'A1': 0.8, 'A2': 0.5, 'B1': 0.1, 'B2': 0.1}, 0.9, 0.2), 
         ({'A1', 'A2', 'B'}, {'B3'}, 1.4)),
         
        (({'A1': 0.8, 'A2': 0.5, 'B1': 0.1, 'B2': 0.01}, 0.9, 0.1), 
         ({'A', 'B1', 'B2'}, {'A3', 'A4'}, 1.3)),
         
        (({'A1': 0.8, 'A2': 0.5, 'B1': 0.1, 'B2': 0.1}, 1.1, 0.2), 
         ({'A', 'B'}, {'A3', 'A4', 'B3'}, 1.56)),
         
        (({'A1': 0.8, 'A2': 0.5, 'B1': 0.1, 'B2': 0.1}, 0.9, 0.1), 
         ({'root'}, {'A3', 'A4', 'B3', 'C'}, 1.31)),
         
        (({'A1': 0.1, 'A2': 0.01, 'B1': 0.8, 'B2': 0.5}, 0.9, 0.1), 
         ({'A1', 'A2', 'B'}, {'B3'}, 1.2)),
         
        (({'A1': 0.1, 'A2': 0.1, 'B1': 0.8, 'B2': 0.5}, 0.9, 0.1), 
         ({'root'}, {'A3', 'A4', 'B3', 'C'}, 1.23)),
         
        (({'A1': 0.1, 'A2': 0.01, 'B1': 0.8, 'B2': 0.5}, 0.9, 0.2), 
         ({'A1', 'A2', 'B'}, {'B3'}, 1.3)),
    ]
    
    for (u_dict, gamma, lmbda), (H, L, p) in data:
        
        tax = PGFS(simple_tax, lmbda, gamma).lift(u_dict)
        assert tax.H == H, tax
        assert tax.L == L, tax
        assert round(tax.p, 2) ==  p, tax

