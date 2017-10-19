#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 19:42:25 2017

@author: hsadeghi
"""
















#%%

#
#def target_sum(T, nums):
#    
#    
#    s = 0
#    
#    for num in nums:
#        s += max(num, -num)
#    
#    if T > s or T < -s:
#        return 0
#    
#    n = len(nums)
#    
#    memo = []
#    for _ in range(n):
#        memo.append({})
#        
#    memo[0][nums[0]] = 1
#    memo[0][-nums[0]] = 1
#    
#    for i in range(1, n):
#
#        for prev in memo[i-1].keys():
#            
#            if prev + nums[i] in memo[i].keys():
#                memo[i][prev + nums[i]] += memo[i-1][prev]
#            else:
#                memo[i][prev + nums[i]] = 1
#                
#            if prev - nums[i] in memo[i].keys():
#                memo[i][prev - nums[i]] += memo[i-1][prev]
#            else:
#                memo[i][prev - nums[i]] = 1
#        
#
#    if T in memo[n-1].keys():
#        return memo[n-1][T], memo
#    else:
#        return 0
#    
#    
#    
#    
#print(target_sum(5, [1,2,-1,3,8]))



#%%
#def fib(n, memo=[]):
#    if n<2:
#        return n
#    if memo == []:
##        global memo
#        memo = [-1] * (n+1)
#        memo[0] = 0
#        memo[1] = 1
#        
#    if memo[n] == -1:
#        memo[n]= fib(n - 1, memo) + fib(n - 2, memo)   
#        
#    return memo[n]
#
#x = 0
#print('fib({}) = '.format(x), fib(x))

#%%
#def fib(n):
#    if n<2:
#        return n
#    global memo
#    memo = [-1] * n
#    memo[0] = 0
#    memo[1] = 1
#    return fibo(n - 1) + fibo(n - 2)
#
#def fibo(n):
#    global memo
#    if memo[n] == -1:
#        memo[n]= fibo(n - 1) + fibo(n - 2)
#        
#    return memo[n]
#
#x = 971
#print('fib({}) = '.format(x), fib(x))


#%%
#class dbl_lnkd_list:
#    
#    def __init__(self, init_list = []):
#        if len(init_list) == 0:
#            self.head = node()
#        else:
#            self.head = node(init_list[0])    
#        cur = self.head    
#        for i in range(len(init_list) - 1):
#            cur.nxt = node(key = init_list[ i + 1], prv = cur.key)
#            cur = cur.nxt
#            
#            
#    ##############################
#class node:
#    def __init__(self, key =None, prv = None, nxt = None ):
#        self.key = key
#        self.prv = prv_node(self, prv)
#        self.nxt = nxt_node(self, nxt)
#        
#    def __repr__(self):
#        return str(self.key)
#         
#        
#class prv_node(node):
#    def __init__(self, nxt_node, key = None):
#        self.key = key
#        self.prv = None
#        self.nxt = nxt_node     
#        
#class nxt_node(node):
#    def __init__(self, prv_node, key = None):
#        self.key = key
#        self.prv = prv_node
#        self.nxt = None 
#
#
##%%
##
#obj_1 = dbl_lnkd_list([1,2,3,4,5])
##
#print(repr(obj_1.head.nxt))

#%%

#def longest_subseq(A, B):
#    
#    m = len(A)
#    n = len(B)
#    
#    memo = [[0] * n] * m  #n x m
#    seq = []
#    
#    if A[0] == B[0]:
#        memo[0][0] = 1
#        seq.append( A[0] )
#        
#    ###First col
#    for row in range(1, m):
#        if A[row] == B[0]:
#            memo[row][0] = 1
#
#    for i in range(len(memo)):
#        print(memo[i][:])
#            
#    # First row    
#    for col in range(1, n):
#        if A[0] == B[col]:
#            memo[col][0] = 1
#            
##    for i in range(len(memo)):
##        print(memo[i][:])
#            
##    for i in range(len(memo)):
##        print(memo[i])
#            
##    for row in range(1, m):
##        for col in range(1, n):
##            if A[row] == B[col]:
##                memo[row][col] = memo[row - 1][col - 1] + 1
##                seq.append( A[row] )
##                print(row, col)
##            else:
##                memo[row][col] = max( memo[row - 1][col], memo[row][col - 1])
#    
#    # finding the largest sequence
#
#    return memo, seq)
#            
#    
#M, S =  longest_subseq('baba', 'abab') 

#print('S=', S)
#
#for i in range(len(M)):
#    print(M[i])




#%%
#def min_coin(n, coins):
#    
#    memo = [0] * (n + 1)    
#    coin_list = [0]
#    for i in range(1, n + 1):
#
#        min_value = 2 ** 20 
#        temp_coin = -1    
#        for coin in coins:    
#            if i - coin >= 0:
#                if min_value > memo[i - coin]:
#                    min_value = memo[i - coin]
#                    temp_coin = coin
#        coin_list.append(temp_coin)
#        memo[i] = min_value + 1
#    
#    
#    final_list = []
#    cur_ind = n
#    while cur_ind - coin_list[cur_ind] > 0 :
#        final_list.append(coin_list[cur_ind]) 
#        cur_ind -= coin_list[cur_ind]
#        
#    final_list.append(coin_list[cur_ind])        
#    return memo[ n ], final_list
#
#print(min_coin(3, [1,5,12,25]))
#n = 3
#memo =  [ 0 1 2 3 0 0 0]
#coin_list = [0]
#
#i = 3
#coin = 1
#min_value = 3
#temp_coin = 1
#
#coin _list = [1, 1, 1]
#final_list = [1, 1 , 1 ]
#cur_ind = 0
