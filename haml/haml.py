import re

from abc       import ABC, abstractmethod
from itertools import combinations, product as cart_prod
from math      import prod
from operator  import methodcaller

import numpy as np


SPLIT_SYM = re.compile(r'{{|\|\||}}')
WEIGHTED = re.compile(r'\s*(\d*\.?\d+)%(.*)', flags=re.DOTALL)
MULTIPLE = re.compile(r'\s*(\d+)-(\d+)%(.*)', flags=re.DOTALL)
RANDVAR  = re.compile(r'%\s*([a-z]+)\((.*)\)%', flags=re.DOTALL)

RANDOM_VALUE_LIMIT = 1


class HAMLObject(ABC):
    @abstractmethod
    def random(self, random_state=None):
        return NotImplemented

    @abstractmethod
    def all(self, random_state=None):
        return NotImplemented
    
    @abstractmethod
    def num_combinations(self):
        return NotImplemented


class HAMLSequence(HAMLObject):
    def __init__(self, objects=None):
        self.objects = objects

    def __repr__(self):
        return self.objects.__repr__()

    def append(self, object):
        self.objects.append(object)

    def random(self, random_state=None):
        rng = np.random.default_rng(random_state)
        return ''.join(obj.random(rng) for obj in self.objects)

    def all(self, random_state=None):
        for strings in cart_prod(*[obj.all() for obj in self.objects]):
            yield ''.join(strings)

    def num_combinations(self):
        return prod([obj.num_combinations() for obj in self.objects])


class HAMLString(HAMLObject):
    def __init__(self, s):
        self.s = s

    def __repr__(self):
        return self.s

    def random(self, random_state=None):
        return self.s

    def all(self, random_state=None):
        yield self.s

    def num_combinations(self):
        return 1


class WeightedChoiceList(HAMLObject):
    def __init__(self, items=None, weights=None):
        self.items = items or []
        self.weights = [1]*len(self.items) if weights is None else weights
        assert len(self.items) == len(self.weights)

    def __repr__(self):
        return 'WCL'+list(zip(self.items, self.weights)).__repr__()

    def add_item(self, item, weight=1):
        assert weight > 0
        self.items.append(item)
        self.weights.append(weight)

    def random(self, random_state=None):
        rng = np.random.default_rng(random_state)
        p = np.asarray(self.weights, dtype=np.float64)
        p = p/p.sum()
        i = rng.choice(len(self.items), p=p)
        return self.items[i].random(rng)

    def all(self, random_state=None):
        for item in self.items:
            yield from item.all()

    def num_combinations(self):
        return sum(item.num_combinations() for item in self.items)


class RandomSubsetList(HAMLObject):
    def __init__(self, items=None, min=0, max=-1):
        self.items = items or []
        self.min = min
        self.max = max

    def __repr__(self):
        return f'RSL({self.min}-{self.max})'+list(self.items).__repr__()
    
    def add_item(self, item):
        self.items.append(item)

    def random(self, random_state=None):
        rng = np.random.default_rng(random_state)
        size = rng.integers(self.min, self.max+1)
        ixs = rng.choice(len(self.items), size=size, replace=False)
        return HAMLSequence(objects=[self.items[i] for i in sorted(ixs)]).random(rng)

    def all(self, random_state=None):
        for size in range(self.min, self.max+1):
            for combo in combinations(self.items, r=size):
                obj = HAMLSequence(objects=list(combo))
                yield from obj.all()

    def num_combinations(self):
        res = 0
        alls = [item.num_combinations() for item in self.items]
        for size in range(self.min, self.max+1):
            for combo in combinations(alls, r=size):
                res += prod(combo)
        return res


class RandomValue(HAMLObject):
    def __init__(self, distribution='normal', **kwargs):
        self.distribution = distribution
        self.kwargs = kwargs

    def __repr__(self):
        return f'RV({self.distribution}; {", ".join(k+"="+v.__repr__() for k,v in self.kwargs.items())})'
    
    def random(self, random_state=None):
        rng = np.random.default_rng(random_state)
        return str(methodcaller(self.distribution, **self.kwargs)(rng))
    
    def all(self, random_state=None):
        global RANDOM_VALUE_LIMIT
        f = methodcaller(self.distribution, **self.kwargs)
        rng = np.random.default_rng(random_state)
        for _ in range(RANDOM_VALUE_LIMIT):
            yield str(f(rng))

    def num_combinations(self):
        global RANDOM_VALUE_LIMIT
        return RANDOM_VALUE_LIMIT



def parse(s: str) -> HAMLObject:
    before, *sections = SPLIT_SYM.split(s)
    level = 0

    result = HAMLSequence([HAMLString(before)])

    items = []
    current_item = []
    for sym, after in zip(SPLIT_SYM.findall(s), sections):
        if sym == '{{':
            level += 1
        elif sym == '}}':
            level -= 1
        
        if level > 0:
            if level == 1 and sym == '||':
                items.append(''.join(current_item[1:]))
                current_item.clear()
            current_item.append(sym)
            current_item.append(after)
        
        elif level == 0:
            items.append(''.join(current_item[1:]))
            current_item.clear()

            # check if the last item is a random variable
            if match := RANDVAR.match(items[-1]):
                dist_name, kwargs = match.groups()
                kwargs_ = {}
                for kwarg in kwargs.split(','):
                    key, val = kwarg.split('=')
                    try:
                        val_ = int(val)
                    except ValueError:
                        val_ = float(val)
                    kwargs_[key.strip()] = val_
                obj = RandomValue(dist_name, **kwargs_)

            elif match := MULTIPLE.match(items[0]):
                # Random Subset List
                min_, max_, item_ = match.groups()
                obj = RandomSubsetList(min=int(min_), max=int(max_))
                obj.add_item(parse(item_))
                for item in items[1:]:
                    obj.add_item(parse(item))
                
            else:
                # Weighted Choice List
                obj = WeightedChoiceList()
                for item in items:
                    if match := WEIGHTED.match(item):
                        groups = match.groups()
                        item_, w = groups[1], float(groups[0])
                    else:
                        item_, w = item, 1.0
                    obj.add_item(parse(item_), w)

            result.append(obj)
            result.append(HAMLString(after))
            items.clear()

        else:
            raise Exception('Parse Error: Misplaced }}')
        
    return result


def parse_file(filename: str):
    with open(filename, 'r') as f:
        content = f.read()
    return parse(content)