# https://twitter.com/karpathy/status/1333217287155847169/photo/1

"""
Say where you live, 1 in 1,000 actively have covid-19.
You feel fatigued and have a slight sore throat, so you take a test, get a positive result.

You learn the test has a 1% false positives, and 10% false negatives.
What's your best guess for your chances of having covid-19?

"""
from random import random, seed
import math
seed(0)

population = 10000000 # 10M
counts = {}

for i in range(population):
    has_covid = i % 1000 == 0 # one in 1000 people have covid

    # assume (big assume) that every person gets tested regardless of any symptoms
    if has_covid:
        tests_positive = True
        if random() < 0.1:  # coin flip create false negative
            tests_positive = False
    else:
        tests_positive = False
        if random() < 0.01:  # coin flip create false positive
            tests_positive = True

    outcome = (has_covid, tests_positive)
    counts[outcome] = counts.get(outcome, 0) + 1

for (has_covid, tests_positive), n in counts.items():
    print(f"Has Covid: {has_covid}, Test Positive: {tests_positive}, Count: {n}")

n_positive = counts[(True, True)] + counts[(False, True)]

print((f"Number of positive tested people: {n_positive}"))
print((f"Probability of having Covid when tested positive : {math.ceil((100.0 * (counts[(True, True)] / n_positive)))}%"))
