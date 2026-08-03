from itertools import combinations_with_replacement
from math import isqrt, prod, gcd
from typing import Generator


def divisors(n):
    d = set()
    for i in range(1, isqrt(n) + 1):
        if n % i == 0:
            d.add(i)
            d.add(n // i)
    return sorted(d)


def best_factors(x, k):

    target = x // 1
    divs = [d for d in divisors(target) if d > 1]

    best = None
    best_score = float("inf")

    for factors in combinations_with_replacement(divs, k):
        if prod(factors) == target:
            score = max(factors) - min(factors)
            if score < best_score:
                best_score = score
                best = factors

    return best


def valid_pairs(x_max: int, y_max: int) -> Generator[tuple[int, int], None, None]:

    for x in range(2, x_max + 3):
        for y in range(2, 2* (y_max + 3), 2):
            d = gcd(x, y)

            if x % (3 * d) == 0:
                value = x // (3 * d)

                if value >= 1:
                    yield x, y