import re
import textwrap
from copy import deepcopy
from io import StringIO

from anytree import LevelOrderIter, Node, find
from ete3 import TextFace, TreeStyle, NodeStyle, RectFace, CircleFace
from pipe import reverse, as_list, as_set


NO_SUPPORT_COLOR = 'Khaki'


def layout(node):
    # Make node name empty for particular nodes
    # if node.name in ('information retrieval', 'information systems'):
    #     node.name = ''
    #     x = 0
    #
    # if node.name == '3 items':
    #     node.name = ''

    # Some long name war here

    try:
        print_label = int(node.e) < 3 or node.Hd == '1' or node.Of == '1' or node.Gap == '1' or node.ForceLabel == '1'


        if print_label:
            name_split = node.name.split('|')
            column = 0
            for line in name_split:

                tw = textwrap.TextWrapper(width=20)
                names = tw.wrap(line)

                for n in names:
                    short_name = TextFace(n, tight_text=True)
                    short_name.rotation = 270
                    node.add_face(short_name, column=column, position="branch-right")
                    column += 1

        # Create and apply node style
        nst = NodeStyle()

        if .4 >= float(node.u) > 0:
            nst["fgcolor"] = "#90ee90"
        elif .6 >= float(node.u) > .4:
            nst["fgcolor"] = "green"
        elif float(node.u) > .6:
            nst["fgcolor"] = "#004000"
        elif node.Gap == '1':
            nst["fgcolor"] = "red"
        else:
            nst["fgcolor"] = NO_SUPPORT_COLOR

        if node.Hd == '1' or node.Of == '1':
            nst["size"] = 40
            nst["shape"] = 'square'
        else:
            nst["size"] = 20
            nst["shape"] = 'circle'

        # if node.Sq == '1' and float(node.u) > 0:
        #     nst["shape"] = 'square'

        node.set_style(nst)

    except:
        print(f'Exception at {node}')


def get_tree_style():
    ts = TreeStyle()
    ts.show_leaf_name = False  # True
    ts.layout_fn = layout
    ts.rotation = 90
    ts.branch_vertical_margin = 10
    ts.show_scale = False
    # ts.mode = "c"
    ts.scale = 50

    ts.title.add_face(TextFace(" ", fsize=20), column=0)

    # for n in t.traverse():
    #     nstyle = NodeStyle()
    #     nstyle["fgcolor"] = "red"
    #     nstyle["size"] = 15
    #     n.set_style(nstyle)

    # ts.show_leaf_name = True
    ts.legend.add_face(TextFace("  "), column=0)
    ts.legend.add_face(TextFace("  "), column=1)
    ts.legend.add_face(RectFace(20, 20, NO_SUPPORT_COLOR, NO_SUPPORT_COLOR), column=0)
    ts.legend.add_face(TextFace("  Topic with no support (u(t)=0)"), column=1)
    ts.legend.add_face(TextFace("  "), column=0)
    ts.legend.add_face(TextFace("  "), column=1)
    ts.legend.add_face(RectFace(20, 20, "#90ee90", "#90ee90"), column=0)
    ts.legend.add_face(TextFace("  Topic with minor support 0<u(t)<=0.4"), column=1)
    ts.legend.add_face(TextFace("  "), column=0)
    ts.legend.add_face(TextFace("  "), column=1)
    ts.legend.add_face(RectFace(20, 20, "green", "green"), column=0)
    ts.legend.add_face(TextFace(u"  Topic with medium support 0.4<u(t)<=0.6   "), column=1)
    ts.legend.add_face(TextFace("  "), column=0)
    ts.legend.add_face(TextFace("  "), column=1)
    ts.legend.add_face(RectFace(20, 20, "#004000", "#004000"), column=0)
    ts.legend.add_face(TextFace("  Topic with high support u(t)>0.6"), column=1)
    ts.legend.add_face(TextFace("  "), column=0)
    ts.legend.add_face(TextFace("  "), column=1)
    ts.legend.add_face(CircleFace(10, "red"), column=0)
    ts.legend.add_face(TextFace("  Gap"), column=1)
    ts.legend.add_face(TextFace("  "), column=0)
    ts.legend.add_face(TextFace("  "), column=1)
    ts.legend.add_face(RectFace(40, 40, "#004000", "#004000"), column=0)  # green
    # ts.legend.add_face(CircleFace(15, "green"), column=1)
    ts.legend.add_face(TextFace("  Head subject or offshoot"), column=1)
    ts.legend_position = 4

    # ts.title.add_face(TextFace(" ", fsize=20), column=0)

    return ts


def compress_pgfs_result(root, u_thresh=0.15):
    """  Compress all descendants with u=0 into one node labeled by nodes count.  """
    root = deepcopy(root)

    heads_and_offshoots = root.H | as_set

    for node in (LevelOrderIter(root) | as_list | reverse):
        if node.u == 0:
            for ch in node.children:
                ch.parent = None
        else:
            gaps_or_disjoint = [ch for ch in node.children if ch.u <= u_thresh]
            if len(gaps_or_disjoint) > 0:
                name = '\n'.join([ch.name for ch in gaps_or_disjoint])
                ancestors_names = [n.name for n in node.ancestors] | as_set
                ancestors_names |= {node.name}
                n_u_below_thresh = len([n for n in gaps_or_disjoint if n.u > 0])
                if len(ancestors_names & heads_and_offshoots) > 0:
                    _ = Node(f'{len(gaps_or_disjoint)} gaps', parent=node, u=0.0, contents=name)
                else:
                    name = f'{len(gaps_or_disjoint)} nodes'
                    if n_u_below_thresh > 0:
                        name += f'|  +|{n_u_below_thresh} minor|offshoots'
                    _ = Node(name, parent=node, u=0.0, contents=name)

                for g in gaps_or_disjoint:
                    g.parent = None

    return root


def pgfs_result_to_newick(root):
    """ Convert PGFS result (tree with lifting attributes) to newick format. """

    out = StringIO()

    heads_and_offshoots = root.H
    offshoots = set()
    for n in heads_and_offshoots:
        node = find(root, filter_=lambda x: x.name == n)
        if node and node.is_leaf:
            offshoots.add(n)
    head_subjects = heads_and_offshoots - offshoots

    def traverse(node, colon=True):
        n_children = len(node.children)
        if n_children > 0:
            out.write('(')
            for i, child in enumerate(node.children, 1):
                traverse(child, i != n_children)
            out.write(') ')

        name_str = re.sub(',', '', node.name)
        name_str = re.sub(r'\(.*\)', '', name_str)

        if ' gaps' in node.name:
            out.write(
                f'{name_str} [&&NHX:p=0.0:e={node.depth}' + ':H={}:u=0.0:V=0.0:G={}:L={}:Hd=0:Of=0:Gap=1:Sq=1:ForceLabel=1]'
            )
        elif ' nodes' in node.name:
            out.write(
                f'{name_str} [&&NHX:p=0.0:e={node.depth}' + ':H={}:u=0.0:V=0.0:G={}:L={}:Hd=0:Of=0:Gap=0:Sq=1:ForceLabel=1]'
            )
        elif len(node.name) > 0:

            # def fmt_set(s):
            #     name = '{' + '; '.join(s) + '}'
            #     return name

            node_str = (
                f'{name_str} [&&NHX:'
                f'p={node.p:.3f}:'
                f'e={node.depth}:'
                #                 f'H={fmt_set(node.H)}:'
                f'u={node.u:.3f}:'
                f'V={node.V}:'
                #                 f'G={fmt_set(node.G)}:'
                #                 f'L={fmt_set(node.L)}:'
                f'Hd={int(node.name in head_subjects)}:'
                f'Of={int(node.name in offshoots)}:'
                f'Gap=0:'
                f'ForceLabel=0:'
                f'Sq=0]'
            ).replace(' -- ', '|')

            out.write(node_str)

        if colon: out.write(', ')
        out.write('\n')

    traverse(root, colon=False)
    out.write(';')

    return out.getvalue()
