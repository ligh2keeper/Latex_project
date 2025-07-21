import numpy as np


UNARY_OPERATORS = {
    'round_brackets': (r'\left({}\right)', 20),
    'square_brackets': (r'\left[{}\right]', 15),
    'braces': (r'\left\{{{}\right\}}', 10),
    'abs': (r'\left|{}\right|', 10),
    'int': (r'\int{{{}}}', 10),
    '2int': (r'\iint{{{}}}', 10),
    '3int': (r'\iiint{{{}}}', 10),
    'sqrt': (r'\sqrt{{{}}}', 30),
    'sum': (r'\sum{{{}}}', 30),
    'derivative': (r"{{{}}}'", 10),
    'log': (r'\log{{{}}}', 50),
    'sin': (r'\sin{{{}}}', 50),
    'cos': (r'\cos{{{}}}', 50)
}

BINARY_OPERATORS = {
    'add': (r'{}+{}', 100),
    'mul': (r'{} {}', 100),
    'div': (r'\frac{{{}}}{{{}}}', 50),
    'slash_div': (r'{}/{}', 20),
    'sub': (r'{}-{}', 100),
    'pow': (r'{}^{{{}}}', 20),
    'greater': ('{}>{}', 10),
    'less': ('{}<{}', 10),
    'greater_eq': ('{}\geqslant {}', 10),
    'less_eq': ('{}\leqslant {}', 10),
    'comma': ('{},{}', 10),
    'equ': ('{}={}', 10)
}

SYMBOLS = {
    '0': ('0', 50),
    '1': ('1', 40),
    '2': ('2', 20),
    'x': ('x', 100),
    'y': ('y', 100),
    'z': ('z', 90),
    't': ('t', 50),
    'i': ('i', 50),
    'j': ('j', 40),
    'k': ('k', 40),
    'n': ('n', 40),
    'f': ('f', 30),
    'g': ('g', 30),
    'h': ('h', 30),
    'inf': ('\infty', 10),
    'A': ('A', 30),
    'B': ('B', 30),
    'C': ('C', 30),
    'a': ('a', 30),
    'b': ('b', 30),
    'c': ('c', 30),
    'd': ('d', 30),
    'alpha': (r'\alpha', 20),
    'beta': (r'\beta', 20),
    'gamma': (r'\gamma', 20),
    'lambda': (r'\lambda', 20),
    'partial': (r'\partial', 10),
    'mu': (r'\mu', 20),
    'pi': (r'\pi', 20),
}

def get_unbinary_dist(n_ops):
    D = [[0] * (2 * n_ops + 1)]
    for _ in range(2 * n_ops):
        D.append([1])
    for n in range(1, 2 * n_ops):
        for e in range(1, 2 * n_ops - n + 1):
            D[e].append(D[e - 1][n] + D[e][n - 1] + D[e + 1][n - 1])
    return D

class ExpressionGenerator(object):
    def __init__(self, n_ops):
        self.n_ops = n_ops
        self.un_ops = list(UNARY_OPERATORS.keys())
        self.bin_ops = list(BINARY_OPERATORS.keys())
        self.un_probs = np.array([UNARY_OPERATORS[name][1] for name in self.un_ops])
        self.un_probs = self.un_probs / self.un_probs.sum()
        self.bin_probs = np.array([BINARY_OPERATORS[name][1] for name in self.bin_ops])
        self.bin_probs = self.bin_probs / self.bin_probs.sum()
        self.all_operators = list(BINARY_OPERATORS.keys()) + list(UNARY_OPERATORS.keys())
        self.symbols = list(SYMBOLS.keys())
        self.symbol_probs = np.array([SYMBOLS[name][1] for name in self.symbols])
        self.symbol_probs = self.symbol_probs / self.symbol_probs.sum()
        self.unbinary_tree_dist = get_unbinary_dist(n_ops)
        self.div_idxs = {
            'div': self.bin_ops.index('div'),
            'pow': self.bin_ops.index('pow'),
            'slash_div': self.bin_ops.index('slash_div')
        }
        self.tokens = ['<s>', '</s>', '<pad>'] + self.bin_ops + self.un_ops + self.symbols
        self.id2token = {i: s for i, s in enumerate(self.tokens)}
        self.token2id = {s: i for i, s in self.id2token.items()}

    def internal_node_to_allocate(self, e, n):
        un_probs = np.array([self.unbinary_tree_dist[e - k][n - 1] for k in range(e)])
        un_probs = un_probs / self.unbinary_tree_dist[e][n]
        bin_probs = np.array([self.unbinary_tree_dist[e - k + 1][n - 1] for k in range(e)])
        bin_probs = bin_probs / self.unbinary_tree_dist[e][n]
        all_probs = np.hstack((un_probs, bin_probs))
        k = np.random.choice(2 * e, p=all_probs)
        is_unary = k < e
        return k % e, is_unary

    def get_leaf(self):
        leaf = np.random.choice(self.symbols, p=self.symbol_probs)
        return [leaf]

    def get_prefix_expr(self, n_ops):
        e = 1
        restrict = {'div': 0,  'pow': 1, 'slash_div': 0}
        bin_probs = self.bin_probs.copy()
        tree = [None]
        skipped = 0
        for n in range(n_ops, 0, -1):
            k, is_unary = self.internal_node_to_allocate(e, n)
            if is_unary:
                op = np.random.choice(self.un_ops, p=self.un_probs)
                e = e - k
                a = 1
            else:
                op = np.random.choice(self.bin_ops, p=bin_probs)
                while op in restrict:
                    restrict[op] += 1
                    if restrict[op] <= 2:
                        break
                    else:
                        bin_probs[self.div_idxs[op]] = 0
                        bin_probs = bin_probs / bin_probs.sum()
                        op = np.random.choice(self.bin_ops, p=bin_probs)

                e = e - k + 1
                a = 2
            skipped += k
            pos = [i for i, v in enumerate(tree) if v is None][skipped]

            tree = tree[:pos] + [op] + [None for _ in range(a)] + tree[pos + 1:]

        num_leaves = len([0 for v in tree if v is None])

        leaves = []

        for _ in range(num_leaves):
            leaf = self.get_leaf()
            leaves.append(leaf)

        np.random.shuffle(leaves)

        for i in range(len(tree) - 1, -1, -1):
            if tree[i] is None:
                tree = tree[:i] + leaves.pop() + tree[i + 1:]

        return tree

    def prefix_to_latex(self, exp):
        v = exp[0]
        v_type = None
        if v in UNARY_OPERATORS.keys():
            v_type, a = 1, 1
        elif v in BINARY_OPERATORS.keys():
            v_type, a = 2, 2

        if v_type is not None:
            args = []
            nxt = exp[1:]
            for _ in range(a):
                arg, nxt = self.prefix_to_latex(nxt)
                args.append(arg)
            if v_type == 1:
                to_str = UNARY_OPERATORS[v][0].format(*args)
            elif v_type == 2:
                to_str = BINARY_OPERATORS[v][0].format(*args)
            return to_str, nxt
        else:
            return SYMBOLS[v][0], exp[1:]
