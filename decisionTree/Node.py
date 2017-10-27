'''

@author: joeyqzhou
'''
class Node:
    value = ""
    children = ""
    
    def __init__(self,val,dictionary):
        self.value = val 
        if(isinstance(dictionary, dict)):
            self.children = dictionary.keys()