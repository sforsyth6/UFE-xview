class AverageMeter(object):
    """Computes and stores the average and current value""" 
    def __init__(self):
        self.reset()
                   
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0 

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_factors(x):
    '''Takes an integer and returns a list of factors'''
    factors = []
    for i in range(1, x + 1):
        if x % i == 0:
            factors.append(i)
    return factors
