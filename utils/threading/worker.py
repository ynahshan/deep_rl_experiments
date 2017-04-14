'''
Created on Apr 13, 2017

@author: Yury
'''

from multiprocessing import Process, Queue

class Worker():    
    @staticmethod
    def __worker__(target, args, q):
        if args != None:
            res = target(*args)
        else:
            res = target()
        q.put(res)
    
    def __init__(self, target, args, name=""):
        '''
        Constructor
        '''
        self.name = name
        self.__q = Queue(1)
        self.__p = Process(target=Worker.__worker__, args=(target, args, self.__q))
        
    def start(self):
        self.__p.start()
        
    def join(self):
        self.__p.join()
    
    def get_result(self):
        try:
            res = self.__q.get(block=False)
            return res
        except:
            return None

class WorkersGroup:
    def __init__(self, num_workers, target, args_list=[], args=None):
        if args != None and len(args_list) > 0:
            raise RuntimeError("Both args and args_list arguments where specified")
        elif args != None and len(args_list) == 0:
            args_list = [args for _ in range(num_workers)]
        elif len(args_list) != num_workers:
            raise RuntimeError("args_list size must be equal to number of workers")
        
        self.__workers = []
        for i in range(num_workers):
            self.__workers.append(Worker(target, args=args_list[i], name=("worker%d" % i)))
            
    def run(self):
        for w in self.__workers:
            w.start()
        for w in self.__workers:
            w.join()
        return [w.get_result() for w in self.__workers]
