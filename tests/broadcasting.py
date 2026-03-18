from itertools import zip_longest

def broadcast_shape(shape_a, shape_b):
    result = []
    for a, b in zip_longest(reversed(shape_a), reversed(shape_b), fillvalue=1):
        if a == b or a == 1 or b == 1:
            result.append(max(a, b))
        else:
            raise ValueError("Shapes not broadcastable")
    return tuple(reversed(result))
print(broadcast_shape((2,3),(3,)))
