class Parent():
    def __init__(self):
        pass
    
    def mainprocess(self):
        print('in parent')
        self._subprocess()
        
    def _subprocess(self):
        print('im parent sub')
        
class Child(Parent):
    def __init__(self):
        super().__init__()
        print('now back to child')
    
    def _subprocess(self):
        print('im child sub')
        
if __name__ == '__main__':
    switch = False
    classss = Parent if switch else Child
    
    instance = classss()
    print(instance.mainprocess())