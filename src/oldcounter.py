# very incomplete substitute for collections.Couter
class Counter:
    def __init__(self,init={}):
        self.data=dict(init)
    def __getitem__(self,key):
        if key in self.data:
            return self.data[key]
        return 0
    def __setitem__(self,key,value):
        self.data[key]=value
    def __iter__(self):
        return self.data.__iter__()
    def values(self):
        for key in self.data:
            yield self.data[key]

if __name__=='__main__':
    c=Counter({2:5})
    c[5]=6
    c[7]=8
    print(c)
    print(sum(c.values()))
    print(sum(c))
