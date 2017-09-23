#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 12:25:07 2017

@author: hsadeghi
"""


from time import time
import numpy as np

import matplotlib.pyplot as plt


#%%

#def min_coin(n, coins):
#    
#    m = len(coins)
#    memo = [0] * (n + 1)    
#
#    for i in range(1, n + 1):
#
#        for j in range(m):
#            min_value = -1 
#            if i - value[j] >= 0:
#                min_value = min( min_value, memo[i - value[j]]) + 1
#        memo[i] = min_value
#            
#    return memo[ n + 1]
#
#print(min_coin(6, [1,5,12,35]))



#%%

    # c[n] =  min_{i = 1, ..., m} {c[n-value[i]]+1} for n>min(coins)
    
    #               0   1   2   3   4   5  6   7   8  ...
    #  coins[0]         1   1
    #  coins[1] 
    #  ...
    #  ...
    
    # path (coin ind)   1    1    1   1   1    1   1
    
#%%
#def perms_string(S):
#
#    n = len(S)
#    
#    if n == 0:
#        print('String should not be empty!')
#        return S
#    
#    list_ = S[0]
#    
#    for i in range(1, n):
#    
#        temp = []
#        
#        for cur_ele in list_:
#
#            for j in range(i + 1):
#            
#                temp.append( cur_ele[0:j] + S[i] + cur_ele[j:n])
#        list_ = temp
#    return list_


#print(perms_string('ab'))


(#%%
#def perms_list(A):
#    n = len(A)    
#    if n == 0:
#        print('List should not be empty!')
#        return A
#    list_ = [ [A[0]] ]
#    
#    for i in range(1, n):    
#        temp = []        
#        for cur_ele in list_:            
#            for j in range(i + 1):            
#                temp.append( cur_ele[0:j] + [A[i]] + cur_ele[j:n])
#        list_ = temp
#        
#    print('number of cases', len(list_))
#    return list_
#
#
#print(perms_list([1,2,3,4]))


#%%

#def binary_rep(x):
#    
#    x_b_raw = bin(abs(x))
#    
#    if x < 0:
#        x_bin = str('-') + x_b_raw[2:]
#        
#    else:
#        x_bin = str('+') + x_b_raw[2:]
#        
#        
#    return x_bin
#
#q = map(binary_rep, [6, -1, 2])
#
#print([i for i in q])
#
#print( binary_rep(-3) )
#print('6 ', binary_rep(6))



#%%
#import queue
#a = queue.Queue()
#
#
#g = (a.put(i) for i in range(5)) 
#
#next(g)
#next(g)
#next(g)
#next(g)
#
#print(a.get())
#print(a.get())
#print(a.get())

#%%
#class node:
#    def __init__(self, value = None, next_ = None):
#        self.value = value
#        self.next_ = next_
#
#    def insert(self, value):
#        new_node = node(value, None)
#        self.next_ = new_node
#        
#        
#
#def has_loop(A):
#    if len(A)==0:
#        return False
#    
#    slow = node(A[0].value, A[0].next)
#    fast = node(A[1].value, A[1].next)
#    
#    while slow.next != None or fast.next != None:
#        if slow.value == fast.value:
#            return True
#        slow =  slow.next
#        fast = fast.next.next
#            
#        
#    return False
    
#%%
#def rmndr_a_pow_b_by_c(a, b , c):
#    
#    res = 1
#    y = a
#    
#    while b > 0:
#        
#        if b % 2 == 1:
#            res =  (res * a) % c 
#            
#        res  =  (res * (a ** 2)) % c
#        
#        b /= 2

#%%
#g = (2**x for x in range(100))
#
#print( next(g) )
#print( next(g) )
#print( next(g) )
#print( next(g) )
#print( next(g) )
#print( next(g) )
#print( next(g) )
#print( next(g) )


#%%
#
#class node():
#    
#    left  = None
#    right = None
# 
#    def __init__(self, value):
#        self.data = value  
#        
#    def __str__(self):
#        return str(self.data)
#        
#tree_1 = node(2)
#
#print(tree_1.left)


#%%








#%%

#def num_rmvd_chars_to_make_anagram(A, B):
#    len_A = len(A)
#    len_B = len(B)
##    
#    if len(A)<=len(B):
#        A_ = A
#        A = B
#        B = A_ 
#          
#    # from now on B is of smaller length
#    H_A = {}
#    for i in range(len_A):
#        if H_A.has_key(A[i]):
#            H_A[A[i]] += 1
#        else:
#            H_A [A[i]] = 1
#                   
#    for i in range(len_B):
#        if H_A.has_key(B[i]):
#            H_A[B[i]] -= 1
#          
#    num_common_chars = len_A - sum(H_A.values())
#    print(num_common_chars)
#    num_remvd_chars = len_A + len_B - 2 * num_common_chars
#    return num_remvd_chars
            
    
#%%
#
#A = "mellion"
#B = "Elliot"
#
#print(num_rmvd_chars_to_make_anagram(A, B))
#

#%%
##def generator(n):
#    for i in range(n):
#        yield  i*i
#
#g = generator(4)
#
#for i in g:
#    print(i)


#%%

#def print_dic(height, weight ):
#    
#    print( height, weight )
#    
#
#
#a={'height':180, 'weight':88}
#
#print_dic(**a)







#%%
#def largest_palindrome():
#    for i in range(0,899):
#        for j in range(0, i+1):
#            first_num = 999 - i
#            second_num = 999 - i + j
#            prod_ = first_num * second_num
#            if is_palindrome(prod_):
#                return prod_, first_num, second_num
#    print('there is no palindrome!')
#        
#    
#def is_palindrome(x):
#    x_str = str(x)
#    len_x = len(x_str)
#    for i in range( int(len_x / 2) ):
#            if x_str[i] != x_str[len_x - i - 1]:
#                return False
#    return True
#
#
#print(largest_palindrome())
                

#globvar = 10
#def read1():
#    print(globvar)
#def write1():
#    global globvar
#    globvar = 5
#def write2():
#    global globvar
#    globvar = 15
#
#read1()
#write1()
#read1()
#write2()
#read1()

#%%
#def next_prime(p):
#    if p == 2:
#        return 3
#    else:
#        cur_ = p + 2
#        while not is_prime(cur_):
#            cur_ = cur_ + 2
#        return cur_
#        
#
#
#cur_ = [5]
#
#for i in range(50):
#    cur_.append (next_prime(cur_[i]))

#plt.plot(cur_)
    
#%%



#def is_prime_0(x):
#    if x % 2 == 0:
##        print('divisibile by', 2)
#        return False        
#    for i in range(3, int(x ** 0.5) + 1, 2):
#        if is_prime(i):
#            if x % i == 0 :
##                print('divisibile by', i)
#                return False
#    return True

#def is_prime_1(x):
#    if x % 2 == 0:
#        print('divisibile by', 2)
#        return False        
#    for i in range(3, int(x ** 0.5) + 1, 2):
#        if x % i == 0 :
#            print('divisibile by', i)
#            return False
#    return True
#        
#start_1 = time()
#print(is_prime_0(67280421310721))
#print('time elapsed=', time()-start_1)
#
#start_2 = time()
#print(is_prime_1(67280421310721))
#print('time elapsed=', time()-start_2)

#%%
#def list_prime_less_than(x):
#    if x<3:
#        raise ValueError('input has to be greater than 2')
#    prime_list = [2]
#    for p_pot in range(3, x, 2):
#        if is_prime(p_pot):
#            prime_list.append(p_pot)
#    return prime_list
#
##%%
#print(list_prime_less_than(11))








#%%
#a = np.random.normal(size=[2**25]) 
#
#start = time()
#b= sorted(a)
#print('elapsed time {} seconds'.format(time()-start)

#%%
#class parent():
#    '''this is an awesome class'''
#    parent_count = 0
#    __parent_count = 0
#    
#    def __init__(self, name, age):
#        self.name = name
#        self.age = age
#        parent.__parent_count +=  1
#    
#    def display_attrs(self):
#        print('age=', self.age)
#        print('name= ' , self.name)
#        print('#parents', self.parent_count)
#        
#    def __call__(self):
#        print(self.__repr__())
#        
#    def __str__(self):
#        return 'object name is %s' % self.name
#        
#    def __repr__(self):
#        return 'object name is %s and their age is %d' %(self.name, self.age)
#    
#    def __cmp__(self,x):
#        return isinstance(x, parent)
#        
#        
##    def __del__(self):
##        class_name = self.name #__class__.__name__
##        print(class_name, "destroyed")
###        print('object {} was deleted... Good bye'.format(self.name))
#
##%%
#class child(parent):
#    
#    def __init__(self, name, age):
#        self.name =  name
#        self.age = age
#             
#    def display_attrs(self):
#        print(self.parent_count)
#
##%% Test
#dad = parent('mahdi', 3245)
##mom = parent('sakine', 23) 
#
#
#print(dad.parent_count)
#setattr(dad, 'parent_count', 1000)
#print(dad.parent_count)
#print(mom.parent_count)
#
#hamed = child('hamed', 29)
#print(hamed._parent__parent_count)

##print(str(dad))
##print(dad)
#
##print(repr(dad))
#
#print(dad)
#print(dad.__cmp__(hamed))





#%%
#class vector():
#    def __init__(self, x, y):
#        self.x = x
#        self.y = y
##        print(str(self))
#       
#    def __str__(self):
#        return 'vector (%d, %d)' %(self.x, self.y)
#    
#    def __add__(self, o):
#        return vector(self.x + o.x, self.y + o.y)
#    
#    
#v1 = vector(1,2)
#
#v2=vector(4, 3)
#v= v1+v2
##print(v1+v2)
#
#print(v)

#%%
#def adder(*args):
#    result = 0
#    for i in args:
#        try:
#            result += i
#        except TypeError:
#            print('uinout has to be a number')
#            break
#            
#        
#        
#    return result
#
##%%
#print( adder(2,3.45656,'hamed'))
