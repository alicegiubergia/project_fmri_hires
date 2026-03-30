#######################################################################
# Toposort implementation for offline environments
#######################################################################
from functools import reduce

def toposort(data):
    """Specific implementation of topological sort for the memory wrapper."""
    data = dict(data)
    for k, v in data.items():
        data[k] = set(v)
    for item in list(data.keys()):
        if item not in data: data[item] = set()
    while True:
        ordered = set(item for item, dep in data.items() if not dep)
        if not ordered:
            break
        yield sorted(ordered)
        data = {item: (dep - ordered) for item, dep in data.items() if item not in ordered}
    if data:
        raise ValueError("Cyclic dependencies exist in the graph")

def toposort_flatten(data, sort=True):
    result = []
    for d in toposort(data):
        result.extend(sorted(d) if sort else list(d))
    return result
