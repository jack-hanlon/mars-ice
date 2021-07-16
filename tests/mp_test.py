import multiprocessing as mp
import numpy as np

def prettyprint(A):
    print(A)
    A = np.array([4,5,6])
    print(A)
def spawn(A):
    prettyprint(A)
    print('test!')

if __name__ == '__main__':
    pool = mp.Pool(mp.cpu_count())


    A = np.array([1,2,3])

    for i in range(5):

        pool.apply(spawn,args=(A,))

    pool.close()
