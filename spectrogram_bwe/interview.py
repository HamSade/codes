#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 21:23:58 2017

@author: hsadeghi
"""

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
def create_node(parent = None, eow=False, left_child={}, right_sib = {}):
    return {'eow':{eow}, 'parent':parent, 'left_child':left_child, 'right_sib':right_sib}


#%%
trie = create_node()

trie['left_child'] = create_node(parent = None, right_sib= {'b'}, left_child={'c'})

trie['left_child']['right_sib'] = create_node(parent = None, right_sib= {'c'}, left_child={'b'})
trie['left_child']['right_sib']['right_sib'] = create_node(parent = None, left_child={'a'})

trie['left_child']['left_child']['right_sib'] = create_node()

trie['left_child']['right_sib']['right_sib']['left_child']['right_sib'] = create_node()

print(trie)


