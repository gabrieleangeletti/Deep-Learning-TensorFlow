
def readText(VocabDir):
    f = open(VocabDir,'r')
    lis = []
    for line in f : 
        [_, s ,_] = line.split(' ') 
        lis.append(s); 
    return lis