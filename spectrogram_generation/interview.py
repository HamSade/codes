#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 21:23:58 2017

@author: hsadeghi
"""


#T = {7: [None, 4, 10],
#     4: [7, 3, 5],
#     3: [4, 1, 3.5],
#     5: [4, None, None],
#     1: [3, None, None],
#     3.5 : [3, None, None]
#
#}

import heapq as hq

a = [1, 4, 2, 7, 4, 8, 2, 7]

print(hq.nlargest(1, a))

#hq.heapify(a)
#print(a)

#%%
#def par_chld(par, chld, which_sib, left = None, right = None):
#    
#    chld = {'key':  par[which_sib]}
#    chld['right'] = right
#    chld['left'] = left
#    par[which_sib]= chld
#    chld['parent'] = par
#    
#    return par,chld
#
#T = {'parent' :  None, 'key' : 7, 'left' : 4, 'right' : 9}
#
#a1 = {}
#a2 = {}
#T, a1 = par_chld(T, a1, 'left', left=3, right=5)
#T, a2 = par_chld(T, a2, 'right', left=8, right=10)
#
#
#a3 = {}
#a1, a3 = par_chld(a1, a3, 'left')
#
#
#print('T=', T)
#
##min finding
##a = T
##while a != None:
##    prev = a
##    a = a['left']
##print('min = ', prev['key'])
#
##max finding
#prev = None
#a = T
#counter = 0
#while 1:
#    print('counter=', counter)
#    prev = a
#    print('prev=', prev)
#    if a:
#        a = a['right']
#    else:
#        break
#    counter += 1
#print('max = ', prev)

#print('min = ', T['left']['left']['key'])

#%%

#import threading as th
#
#class ham_class(th.Thread):
#    def run(self):
#        for _ in range(10):
#            print(th.currentThread().getName())
#            
#            
# 
#obj1 = ham_class(name= 'hamed')
#obj2 = ham_class(name='david')
#
#obj1.start()
#
#obj2.start()




#%%

#Tries
#
#def create_node(parent = None, eow=False, left_child={}, right_sib = {}):
#    return {'eow':{eow}, 'parent':parent, 'left_child':left_child, 'right_sib':right_sib}
#
#
##%%
#trie = create_node()
#
#trie['left_child'] = create_node(parent = None, right_sib= {'b'}, left_child={'c'})
#
#trie['left_child']['right_sib'] = create_node(parent = None, right_sib= {'c'}, left_child={'b'})
#trie['left_child']['right_sib']['right_sib'] = create_node(parent = None, left_child={'a'})
#
#trie['left_child']['left_child']['right_sib'] = create_node()
#
#trie['left_child']['right_sib']['right_sib']['left_child']['right_sib'] = create_node()
#
#print(trie)


