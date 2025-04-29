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


def split_top_level(path, delims='[;]'):
    level = 0
    items = []
    for i, char in enumerate(path):
        if char == delims[0]:
            level += 1
            if level == 1:
                item_start = i+1
        elif char == delims[-1]:
            level -= 1
            if level == 0:
                item_end = i
        if level == 1 and char in delims[1:-1]:
            item_end = i
            items.append(path[item_start:item_end])
            item_start = i+1
    return items
            


class HAMLObject(ABC):
    
    @abstractmethod
    def __getitem__(self, path):
        return NotImplemented

    @abstractmethod
    def random(self, return_path=False, random_state=None):
        return NotImplemented

    @abstractmethod
    def all(self, return_path=False, random_state=None):
        return NotImplemented
    
    @abstractmethod
    def num_combinations(self):
        return NotImplemented


class HAMLSequence(HAMLObject):
    def __init__(self, objects=None):
        self.objects = objects

    @property
    def non_string_objects(self):
        return [obj for obj in self.objects if not isinstance(obj, HAMLString)]

    def __repr__(self):
        return self.objects.__repr__()

    def __getitem__(self, path):
        return ''.join(obj[p] for obj, p in zip(self.non_string_objects, split_top_level(path)))

    def append(self, object):
        self.objects.append(object)

    def random(self, return_path=False, random_state=None):
        elems = [obj.random(return_path, random_state) for obj in self.objects]
        if return_path:
            elems, paths = zip(*elems)
            path = '['+';'.join(p for p in paths if p)+']'
            return ''.join(elems), path
        return ''.join(elems)

    def all(self, return_path=False, random_state=None):
        for strings in cart_prod(*[obj.all(return_path, random_state) for obj in self.objects]):
            if return_path:
                strings, paths = zip(*strings)
                path = '['+';'.join(p for p in paths if p)+']'
                yield ''.join(strings), path
            else:
                yield ''.join(strings)

    def num_combinations(self):
        return prod([obj.num_combinations() for obj in self.objects])


class HAMLString(HAMLObject):
    def __init__(self, s):
        self.s = s

    def __repr__(self):
        return self.s
    
    def __getitem__(self, key):
        return self.s

    def random(self, return_path=False, random_state=None):
        return (self.s, '') if return_path else self.s

    def all(self, return_path=False, random_state=None):
        yield (self.s, '') if return_path else self.s

    def num_combinations(self):
        return 1


class WeightedChoiceList(HAMLObject):
    def __init__(self, items=None, weights=None):
        self.items = items or []
        self.weights = [1]*len(self.items) if weights is None else weights
        assert len(self.items) == len(self.weights)

    def __repr__(self):
        return 'WCL'+list(zip(self.items, self.weights)).__repr__()

    def __getitem__(self, path):
        try:
            ix, path_ = path.split('>', 1)
            return self.items[int(ix)][path_]
        except ValueError:
            return self.items[int(path)]

    def add_item(self, item, weight=1):
        assert weight > 0
        self.items.append(item)
        self.weights.append(weight)

    def random(self, return_path=False, random_state=None):
        rng = np.random.default_rng(random_state)
        p = np.asarray(self.weights, dtype=np.float64)
        p = p/p.sum()
        i = rng.choice(len(self.items), p=p)
        obj = self.items[i].random(return_path, rng)
        if return_path:
            obj, path = obj
            path = f'{i}>{path}' if path else f'{i}'
            return obj, path
        return obj

    def all(self, return_path=False, random_state=None):
        for i, item in enumerate(self.items):
            for elem in item.all(return_path, random_state):
                if return_path:
                    elem, path = elem
                    path = f'{i}>{path}' if path else f'{i}'
                    yield elem, path
                else:
                    yield elem

    def num_combinations(self):
        return sum(item.num_combinations() for item in self.items)


class RandomSubsetList(HAMLObject):
    def __init__(self, items=None, min=0, max=-1):
        self.items = items or []
        self.min = min
        self.max = max

    def __repr__(self):
        return f'RSL({self.min}-{self.max})'+list(self.items).__repr__()
    
    def __getitem__(self, path):
        items = []
        for path_ in split_top_level(path, '{;}'):
            try:
                i, path__ = path_.split('>', 1)
                items.append((self.items[int(i)], path__))
            except ValueError:
                items.append((self.items[int(path_)], ''))
        return ''.join(item[p] for item, p in items)

    def add_item(self, item):
        self.items.append(item)

    def random(self, return_path=False, random_state=None):
        rng = np.random.default_rng(random_state)
        size = rng.integers(self.min, self.max+1)
        ixs = rng.choice(len(self.items), size=size, replace=False)
        elems = [self.items[i].random(return_path, rng) for i in sorted(ixs)]
        if return_path:
            elems, paths = zip(*elems)
            paths = [f'{i}>{p}' if p else f'{i}' for i, p in zip(sorted(ixs), paths)]
            path = '{'+';'.join(paths)+'}'
            return ''.join(elems), path
        return ''.join(elems)

    def all(self, return_path=False, random_state=None):
        for size in range(self.min, self.max+1):
            for combo in combinations(range(len(self.items)), r=size):
                for strings in cart_prod(
                        *[self.items[i].all(return_path, random_state) for i in combo]):
                    if return_path:
                        strings, paths = zip(*strings)
                        paths = [f'{i}>{p}' if p else f'{i}' for i,p in zip(combo, paths)]
                        path = '{'+';'.join(paths)+'}'
                        yield ''.join(strings), path
                    else:
                        yield ''.join(strings)

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
    
    def __getitem__(self, path):
        assert path.startswith('%')
        return path[1:]

    def random(self, return_path=False, random_state=None):
        rng = np.random.default_rng(random_state)
        value = str(methodcaller(self.distribution, **self.kwargs)(rng))
        return (value, f'%{value}') if return_path else value
    
    def all(self, return_path=False, random_state=None):
        global RANDOM_VALUE_LIMIT
        f = methodcaller(self.distribution, **self.kwargs)
        rng = np.random.default_rng(random_state)
        for _ in range(RANDOM_VALUE_LIMIT):
            value = str(f(rng))
            yield (value, f'%{value}') if return_path else value

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