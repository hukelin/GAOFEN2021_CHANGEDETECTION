import random
from sklearn.model_selection import train_test_split

x = [i for i in range(1000)]
a, b = train_test_split(x, test_size=0.2, random_state=0)
c, d = train_test_split(a, test_size=len(x) / len(a) * 0.2, random_state=0)
e, f = train_test_split(c, test_size=len(x) / len(c) * 0.2, random_state=0)
g, h = train_test_split(e, test_size=len(x) / len(e) * 0.2, random_state=0)

print(len(b))
print(len(d))
print(len(f))
print(len(h))
print(len(g))
