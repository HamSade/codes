#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 12:25:07 2017

@author: hsadeghi
"""


#from time import time
#import numpy as np
#import matplotlib.pyplot as plt

#%%

adj_list = {}
adj_list [1] = {2:3, 3: 8}
adj_list [2] = {4:6}
adj_list [3] = {1: 8, 2:5}
adj_list [4] = {2: 6}

#print(adj_list)

#%%















#%%
#class node():
#    def __init__(self):
#        self.parent = None
#        self.color = 'white'
#        self.d = 0
#        self.f = 0
#%%
#def DFS_visit(G, node_ind):
#    global time
#    global nodes
#    
#    time += 1
#    nodes[node_ind].d = time 
#    nodes[node_ind].color = 'grey'
#    
#    for adj_node in G[node_ind].keys():
#        if nodes[adj_node].color == 'white':
#            nodes[adj_node].parent = node_ind
#            DFS_visit(G, adj_node)
#            
#    nodes[node_ind].color = 'black'
#    time += 1
#    nodes[node_ind].f = time
#    
##%%      
#def DFS(G, start_ind):
#    
#    V = len(G)  
#    global nodes
#    nodes = [0]
#    
#    for _ in range(V):  #node_ind runs from 1..V
#        nodes.append(node())   
#        
#    global time
#    time = 0
#
#    sorted_ind_list = [i for i in range(1, V + 1)]
#    sorted_ind_list.remove(start_ind)
#    sorted_ind_list.reverse()
#    sorted_ind_list.append(start_ind)
#    sorted_ind_list.reverse()
#    
#    for node_ind in sorted_ind_list:
#        if nodes[node_ind].color == 'white':
#            DFS_visit(G, node_ind)
#    return time, nodes


#%%
tt, nodes_ = DFS(adj_list, 1)
nodes_.remove(0)

print(tt)
counter = 1
for u in nodes_:
    print('node_ind = ', counter, 'color =', u.color,
          'parent = ', u.parent, 'd =' , u.d,\
          'f = ', u.f)
    counter += 1


#%%
#def DFS(G, start_ind):
#    
#    V = len(G)  
#    global attr_
#    attr_ = {}
#    
#    for node_ind in range(1, V + 1):  #node_ind runs from 1..V
#        attr_[node_ind] = {'color': 'white'}
#        attr_[node_ind]['parent'] = None
#        attr_[node_ind]['d'] = 0
#        attr_[node_ind]['f'] = 0
#        
#    global time_
#    time_ = 0
#
#    sorted_ind_list = [i for i in range(1, V + 1)]
#    sorted_ind_list.remove(start_ind)
#    sorted_ind_list.reverse()
#    sorted_ind_list.append(start_ind)
#    sorted_ind_list.reverse()
#    
#    for node_ind in sorted_ind_list:
#        if attr_[node_ind]['color'] == 'White':
#            DFS_visit(G, node_ind)
#            
#    return time_, attr_
#        
##%%
#def DFS_visit(G, node_ind):
#    global time_
#    global attr_
#    
#    time_ += 1
#    attr_[node_ind]['d'] = time_ 
#    attr_[node_ind]['color'] = 'grey'
#    
#    for adj_node in G[node_ind].keys():
#        if attr_[adj_node]['color'] == 'white':
#            attr_[adj_node]['parent'] = node_ind
#            DFS_visit(G, adj_node)
#            
#    attr_[node_ind]['color'] = 'black'
#    time_ += 1
#    attr_[node_ind]['f'] = time_
         
    
#%%
#tt, aa = DFS(adj_list, 1)
#print(tt, aa)     

#%%
#import math

#def BFS(G, start_idx, end_idx):
#    # G is a dictionary of dics (adj list)
#    
#    V = len(G)
#    
#    attr = {start_idx :{'d':0, 'color':'grey'}}
#    queue_nodes = {start_idx}
#    parent_list = [None] * V
#    
#    
#    for node_idx in range(1, V + 1):
#        if node_idx != start_idx:
#            attr[node_idx] = {'color': 'white'}
#            attr[node_idx]['d'] = float('inf')
#    
#    while len(queue_nodes)>0:
#        cur_node = queue_nodes.pop()
#        # going over all adjacent edges
#        
#        
#        for adj_node in G[cur_node].keys(): 
#             if attr[adj_node]['color'] == 'white':
#                 
#                 queue_nodes.add(adj_node)
#                 attr[adj_node]['color'] = 'grey'
#                 
#                 temp_dist = attr[cur_node]['d'] + G[cur_node][adj_node]
#                 if attr[adj_node]['d'] > temp_dist:
#                     parent_list[adj_node - 1] = cur_node
#                     attr[adj_node]['d'] = temp_dist
#                         
#             attr[cur_node]['color'] = 'black'
#        
#        path = []
#        cur_idx = end_idx
#        while cur_idx > 1:
#            path.append(parent_list[cur_idx - 1])
#            cur_idx = parent_list[cur_idx - 1]
#
#    return attr[end_idx]['d'], path
#    
#        
#print(BFS(adj_list, 2, 3))
             
             
        
    
    

    
#%%
#def max_value(n, W, V):
#    m = len(W)
#    memo = [0] * (n + 1)
#    weight_list = [0]
#    for i in range(1, n + 1):
#        max_value = 0
#        temp_weight = 0
#        for j in range(m):
#            if i >= W[j]:
#                rem =  memo[i - W[j]] + V[j]
#                if max_value < rem:
#                    max_value = rem
#                    temp_weight = W[j]
#        memo[i] = max_value
#        weight_list.append(temp_weight)
#        
#    final_list= []
#    cur_ind = n
#    while cur_ind > min(W):
#        final_list.append(weight_list[cur_ind])
#        cur_ind -= weight_list[cur_ind]
#    return memo[n], final_list
#
#print(max_value( 21, [10, 20, 30], [50, 120, 200]))


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


#%%
#friends ={}
#friends['mashal']='khale zanak, khod an pendar'
#friends['baji']='adam froosoh'
#friends['fred']='zan zalil'
#print ([friends[i] for i in friends])
#
#for i in friends:
#    print(friends[i])


#%%

#import heapq
#
#stocks ={'apple': 500,
#         'gogole': 230,
#         'microsoft': 110}
#
#print(stocks.values())

#print(heapq.nlargest(2, stocks,key=stocks.keys()))

#%%

#a= [1,2,3,4]
#
#doubler = lambda x: 2*x
#
#b = list( map(doubler, a) )
#print(b)

