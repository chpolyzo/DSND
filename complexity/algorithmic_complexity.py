# challenge
# Given the following anonymousFurnction()

def anonymousFunction(alist):
    for passnum in range(len(alist)-1, 0, -1):
        for i in range(passnum):
            if alist[i]>alist[i+1]:
                temp = alist[i]
                alist[i] = alist[i+1]
                alist[i+1] = temp


# What is the algorithmic complexity of the anonymousFurnction()?
# [ie: if we give this function] an input of size N, what is the order of
# magnitude, as a function of N, of the number of operations it will perform?]
