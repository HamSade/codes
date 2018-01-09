#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 09:21:33 2017

@author: hsadeghi
"""

#%%
class matrix_graph:
    
    
    def __init__(self, A):
        
        self.A = A
        self.n = len(A[0])
    
        self.G = {}
        for i in range(self.n):
            for j in range(self.n):
                if A[i][j] == 1:
                    self.G = {(i,j): {'status' : "blocked",'pi': None, 'd' : 0, 'f' : 0}}    
                else:
                    self.G = {(i,j): {'status' : "white",'pi': None, 'd':0, 'f':0}}
     
        
    
    def is_blocked(self, src):
        # src is a two-element tuple

        cur_loc = None
        next_loc = src
        
        while next_loc != cur_loc:
            
            cur_loc = next_loc
            next_loc = matrix_graph.visit_neighbors(cur_loc)
        
            if self.A[next_loc[0]][next_loc[1]] == 10:
                return False # there is a path
            
        return True # There is no path, i.e. blocked

###################################
    def visit_neighbors(self, cur_loc):
        "It first visit North, then East, South and West"
        
        row = cur_loc[0]
        col = cur_loc[1]
        self.G[cur_loc]["color"] = "gray"
        
        if row - 1 > 0 and self.A[row - 1][col] == 0 and self.G[(row - 1, col)]['status'] == 'white': #North
            next_loc = (row-1, col)
       
        elif col + 1 < self.n and self.A[row][col + 1] == 0 and self.G[(row, col + 1)]['status'] == 'white': # East
            next_loc = (row, col + 1)
            
        elif row + 1 < self.n and self.A[row - 1][col] == 0 and self.G[(row - 1, col)]['status'] == 'white': # South
            next_loc = (row + 1, col)
            
        elif col - 1 > 0 and self.A[row - 1][col] == 0 and self.G[(row - 1, col)]['status'] == 'white': # West
            next_loc = (row, col - 1)
            
        else:
            next_loc = cur_loc
    
    
        if next_loc != cur_loc:
            next_loc = matrix_graph.visit_neighbors(cur_loc)
                
        else:
            self.G[next_loc]["status"] = "black"
            
        return next_loc
                
        
    