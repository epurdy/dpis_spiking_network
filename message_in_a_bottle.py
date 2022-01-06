"""This file is notionally complete, but is not tested at all and
probably contains serious bugs of mistranslation. It's also very slow
because I translated the for-loops from the C version into python
for-loops.

This file not working, or being extremely slow, should not be held
against Dpi's proposal.
"""

import numpy as np
import tensorflow as tf

NTYPE = 10
INSIZE = 784

# size. NSYN: For 98.9%: 7920. For 98.65% : 792
NSYN = 10

NSIZE = 792 # should be a multiple of the number of threads?

THRESHOLD = 1_000_000
THRESHOLD_MIN = 300_000
THRESHOLD_MAX = 1_200_000
MIN_WEIGHT = 97_000

NCYC = 4

NUM = 60_000 # recurs a lot in the following, don't know what to call
             # it yet

wd = np.array([192, 128, 64, 1])

list_ = np.zeros((3, NUM), dtype=np.int64)
list_size = [NUM, 10_000, 1_000]
list_set = [0, 1, 0]
set_size = [NUM, 10_000]
type_ = np.zeros((2, NUM), dtype=np.uint8)  # seems to be MNIST class labels
in_ = np.zeros((2, NUM, INSIZE), dtype=np.uint8) # seems to be MNIST images
notnuls = np.zeros(NUM, dtype=np.int64)
nbc = np.zeros(NTYPE, dtype=np.int64)
nbcn = np.zeros((NTYPE, NSIZE), dtype=np.int64)
zero = np.zeros((NUM * NTYPE), dtype=np.int64)
err = np.zeros(NUM, dtype=np.int64)
num_spikes = np.zeros((3, NUM, NTYPE), dtype=np.int64) # number of spikes
                                                   # per group and
                                                   # sample
weight = np.zeros((NTYPE, NSIZE, NSYN), dtype=np.int64)
from_ = np.zeros((NTYPE, NSIZE, NSYN), dtype=np.int16)
tmpw = np.zeros((NTYPE * NSIZE * NSYN), dtype=np.int64)
tmpf = np.zeros((NTYPE * NSIZE * NSYN), dtype=np.int64)

quantP = 0.0
quantDivP = 100.0
quantUpP = 19.0
quantN = 0.0
quantDivN = 900.0
quantUpN = 150.0
adjp = 500
adjn = -500
divNew = 10
nbNew = 10
lossw = 20
losswNb = 100


def init():
    for set_ in range(2):
        for l in range(set_size[set_]):
            list_[set_, l] = l

def quant(sample):
    global quantP, quantN
    is_ = type_[0, sample]
    if err[sample] > quantP:
        quantP += quantUpP / quantDivP
    elif err[sample] < quantP:
        quantP -= 1.0 / quantDivP

    tot = num_spikes[2, sample, :].sum()

    for isnot in range(NTYPE):
        if isnot == is_:
            continue

        d = num_spikes[2, sample, isnot] - ((tot - num_spikes[2, sample, isnot])/9)
        if d > quantN:
            quantN += quantUpN / quantDivN
        elif d < quantN:
            quantN -= 1.0 / quantDivN
        
# skipping over all the GPU code for now

# skipping readset as it seems to just load MNIST

def loss(b):
    # move weights by lossw
    frozen_weight = weight.copy()
    if b % losswNb == 0:
        weight[frozen_weight > 0] -= lossw
        weight[frozen_weight < 0] += lossw

    # zero out small weights
    small_nonzero_weights = np.logical_and(weight != 0, weight.abs() < MIN_WEIGHT)
    nbcn -= (small_nonzero_weights).sum(axis=2)
    nbc -= (small_nonzero_weights).sum(axis=2).sum(axis=1)
    weight[small_nonzero_weights] = 0
        
    # for a in range(NTYPE):
    #     for n in range(NSIZE):
    #         for s in range(NSYN):
    #             w = weight[a, n, s]
    #             if w:
    #                 if lossw and (b % losswNb == 0):
    #                     if w > 0:
    #                         w -= lossw
    #                     else:
    #                         w += lossw
    #                 if abs(w) < MIN_WEIGHT:
    #                     w = 0
    #                     nbc[a] -= 1
    #                     nbcn[a, n] -= 1

    #                 weight[a, n, s] = w


def connect(nb, sample, a, init):
    ps = np.zeros(1000, dtype=np.int64)
    pn = 0
    
    if nbc[a] >= NSIZE * NSYN:
        return

    nb = min(nb, NSIZE * NSYN - nbc[a])

    ps = list(range(NSIZE * NSYN - nbc[a]))
    np.random.shuffle(ps)
    ps = ps[:nb]
    ps = sorted(ps)

    n = 0
    s = 0
    p0 = 0

    for i, p in enumerate(ps):
        while n < NSIZE and p0 != p:
            if p0 + NSYN - nbcn[a, n] < p:
                p0 += NSYN - nbcn[a, n]
            else:
                s = s * (pn == n)
                while s < NSYN and p0 != p:
                    if weight[a,n,s] != 0:
                        p0 += 1
                    s += (p0 != p)

            n += (p0 != p)

        n = min(n, NSIZE - 1)
        s = min(s, NSYN - 1)
        
        if notnuls[sample]:
            f = 1 + np.random.randint(notnuls[sample])
        else:
            f = 1

        f0 = 0
        while f0 < INSIZE and f:
            if in_[0, sample, f0]:
                f -= 1
            f0 += (f != 0)

        from_[a, n, s] = f0
        weight[a, n, s] = init
        nbc[a] += 1
        nbcn[a, n] += 1
        pn = n

def test_mt_one(listNb):
    set_ = list_set[listNb]
    for l in range(min(10, list_size[listNb])):
        sample = list_[listNb, l]
        for a in range(NTYPE):
            num_spikes[listNb, sample, a] = 0
            for n in range(NSIZE):
                t = 0
                for c in range(NCYC):
                    for s in range(NSYN):
                        if in_[set_][sample][from_[a, n, s]] >= wd[c]:
                            t += weight[a, n, s]

                    if t > THRESHOLD:
                        num_spikes[listNb, sample, a] += 1
                    else:
                        t = t >> 1

def learn_mt_one(mtsample, mta, mtadj):
    for n in range(NSIZE):
        tot = 0
        cnta = 0
        for c in range(NCYC):
            mask = 1
            for s in range(NSYN):
                if weight[mta, n, s] and in_[0, mtsample, from_[mta, n, s]] > wd[c]:
                    tot += weight[mta, n, s]
                    cnta = cnta | mask
                mask = mask << 1

            if tot > THRESHOLD_MIN:
                if tot < THRESHOLD_MAX:
                    mask = 1
                    for s in range(NSYN):
                        if (cnta & mask) != 0 and weight[mta, n, s]:
                            if abs(weight[mta, n, s] + mtadj) < THRESHOLD:
                                weight[mta, n, s] += mtadj
                                if abs(weight[mta, n, s]) < MIN_WEIGHT:
                                    weight[mta, n, s] = 0
                                    nbcn[mta, n] -= 1

                        mask = mask << 1
                tot = 0
                cnta = 0
            tot = tot >> 1

def learn(sample, a, adjp):
    learn_mt_one(sample, a, adjp)
    nbc[a] = nbcn[a, :].sum()
            

def test(listNb):
    test_mt_one(listNb=listNb)
    set_ = list_set[listNb]
    ok = 0
    for l in range(min(10, list_size[listNb])):
        sample = list_[listNb, l]
        is_ = type_[set_, sample]
        b = -1

        for a in range(NTYPE):
            if (a != is_ and num_spikes[listNb, sample, a] > b):
                b = num_spikes[listNb, sample, a]
                isnot = a

        if num_spikes[listNb, sample, is_] > b:
            ok += 1

        if listNb == 2:
            err[sample] = b - num_spikes[2, sample, is_]
            quant(sample)
        

    return ok
            
def batch():
    sel = np.zeros(NUM, dtype=np.int64)
    d = 0.0

    prev_step = 0
    nb_tested = 0
    nblearn = 0
    bu = 0
    b = 0
    for batch_num in range(1_000_000_000):
        print('batch', batch_num)
        batch_elements = list(range(NUM))
        np.random.shuffle(batch_elements)
        batch_elements = batch_elements[:list_size[2]]
        list_[2][:len(batch_elements)] = batch_elements
        sel[batch_elements] = 1

        listNb = 2;
        test(listNb=2)
        nb_tested += 1

        for l in range(list_size[2]):
            sample = list_[2, l]
            is_ = type_[0, sample]

            if (err[sample] >= quantP or nb_tested > b):
                connect(nbNew, sample, is_, THRESHOLD / divNew)
                learn(sample, is_, adjp)
                #toreload[is_] = 1
                nblearn += 1
                b += 1
                if (lossw != 0 and nblearn % losswNb == 0):
                    loss(b)

            if bu > (3 * b) / 2:
                continue

            tot = num_spikes[2, sample, :].sum()
            for isnot in range(NTYPE):
                if isnot == is_:
                    continue

                d = num_spikes[2, sample, isnot] - ((tot - num_spikes[2, sample, isnot])/9.0)

                if quantUpN != 0.0 and d >= quantN:
                    connect(nbNew, sample, isnot, -THRESHOLD/divNew)
                    learn(sample, isnot, adjn)
                    #toreload[isnot] = 1
                    bu += 1

        if (b != prev_step):
            print("testing", nblearn, ":")
            print("\t0: %2.3f  " % float(test(listNb=0)))
            print("\t1: %2.3f  " % float(test(listNb=1)))
            prev_step = b
            
            


def main():
    np.random.seed(1000)

    # load MNIST
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    in_[0, :, :] = np.reshape(x_train, (-1, INSIZE))
    in_[1, :10_000, :] = np.reshape(x_test, (-1, INSIZE))
    type_[0, :] = y_train
    type_[1, :10_000] = y_test
    
    init()

    brief_step = 20_000
    list_size[2] = 5
    
    batch()

if __name__ == '__main__':
    main()
