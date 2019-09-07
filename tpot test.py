from tpot import TPOTClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

import time

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target,
                                                    train_size=0.80, test_size=0.20)

gen = [5, 10, 15, 20, 25, 30, 50, 100]
pop = [5, 10, 15, 20, 25, 30, 50, 100]

for gens in gen:
  for pops in pop:
    tpot = TPOTClassifier(generations=gens, population_size=pops, verbosity=2)
    start = time.time()
    print('#generations = ', gens, '#population = ', pops)
    tpot.fit(X_train, y_train)
    end = time.time()
    print('time: ',(end-start))
    print('accuracy :', tpot.score(X_test, y_test))
    tpot.export('tpot_mnist_like_digits_gens{}_pops{}_{}.py'.format(gens,pops,tpot.score(X_test, y_test)))