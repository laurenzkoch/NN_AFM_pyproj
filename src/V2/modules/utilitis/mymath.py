from itertools import combinations_with_replacement
from math import isqrt, prod


def divisors(n: int) -> list:
    d = set()
    for i in range(1, isqrt(n) + 1):
        if n % i == 0:
            d.add(i)
            d.add(n // i)
    return sorted(d)


def best_factors(target: int, k: int) -> tuple:

    divs = [d for d in divisors(target)]

    best = (None,)
    best_score = float("inf")

    for factors in combinations_with_replacement(divs, k):
        if prod(factors) == target:
            score = max(factors) - min(factors)
            if score < best_score:
                best_score = score
                best = factors
                best = tuple(sorted(best, reverse=True))
    return best
