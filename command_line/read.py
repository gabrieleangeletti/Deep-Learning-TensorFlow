

text_dir = 
f = open(text_dir,'r')
for line in f:
    n, word, _ = [f.split(" ")]

# for i in range(1, t + 1):
#   n, m = [int(s) for s in input().split(" ")]  # read a list of integers, 2 in this case
#   print("Case #{}: {} {}".format(i, n + m, n * m))
#   # check out .format's specification for more formatting options