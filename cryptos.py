from itertools import permutations

CRYPTO = ['BTC', 'ETH', 'BCH']
CUR = ['EUR', 'USD']

PAIRS = []

for cr in CRYPTO:
    for cur in CUR:
        PAIRS += [cr + cur]
