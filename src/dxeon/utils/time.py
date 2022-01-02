from time import time
import numpy as np

START_ = 0
ACC_ = None

def time_profile(action = 'start', accumulate = False):
    global START_, ACC_, COUNT_

    if action == 'start':
        START_ = time()
        if accumulate:
            del ACC_
            ACC_ = []
        else:
            ACC_ = None

    elif action == 'end':
        took = time() - START_
        START_ = time()
        print(f'Run completed in {took:.5f} secs!')
        if ACC_ is not None:
            ACC_.append(took)
    
    elif action == 'finalize':
        assert ACC_ is not None, f'\n\nSet `accumulate` to `True` to finalize the time profiling.\n'
        acc = np.array(ACC_)
        print('\n[===== Time profiling completed =====]')
        print(f'Mean time taken   : {acc.mean():.5f} secs')
        print(f'Median time taken : {np.median(acc):.5f} secs\n')
