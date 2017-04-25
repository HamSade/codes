#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 15:47:51 2017

@author: hsadeghi
"""

class employee():
    
    
    def __init__(self, first, last):
        self.first = first
        self.last = last
        self.email = self.first +'.'+ self.last + '@email.com'
        
    @property    
    def fullname(self, middle_name):
        return '{} {} {}'.format(self.first, middle_name, self.last)
        

emp_1= employee('hamed', 'sadeghi')

emp_1.first='Jax'

print(emp_1.first)
print(emp_1.last)      
print(emp_1.email)       
print(emp_1.fullname)      

    
    
    