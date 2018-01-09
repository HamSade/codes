#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 19:42:25 2017

@author: hsadeghi
"""




























#%%
#m = 6
#
#def num(x, n):
#    
#    if x < n or x > m * n:
#        return 'x should be within reasonable range'
#        
#    if x == n or n == 1:
#        return 1
#    
#    memo = [[0 for _ in range(n + 1)] for __ in range(x + 1)]
#    
#    for i in range(1, x + 1):
#        memo[i][1] = 1
#        
#    
#    for i in range(1, x + 1):
#        for j in range(1, n + 1):
#            for s in range(1, m + 1):
#                if i - s >= 0:
#                    memo[i][j] += memo[i - s][j - 1]
#    
#    return memo[x][n]
#
#print(num(8, 3))



#%%

#inf = float('inf')
#
#def min_attempts(k, n):
#    if k == 1:
#        return 1
#    elif n == 1:
#        return k
#      
#    memo = [[inf for _ in range(n + 1)] for __ in range(k + 1)]
#    
#    
#    
#    for j in range(n + 1):
#
#        memo[1][j] = 1
#        
#    for i in range(k + 1):
#        memo[i][1] = i
#
#    memo[0][1] = inf
#    
#    for i in range(2, k + 1):
#        for j in range(2, n + 1): 
#            for s in range(1, i):
#                if memo[i][j] > max(memo[i - s][j], memo[s][j - 1]):
#                    memo[i][j] = max(memo[i - s][j], memo[s][j - 1])
#            memo[i][j] += 1 
#    
#    return memo[k][n]
#    
#    
#print(min_attempts(12, 3))


#%%


#def can_reach(xs, ys, xt, yt):
#  
#  if not is_in_range(xs, ys) or not is_in_range(xt, yt):
#    return False
#    
#  if xs == xt and ys == yt:
#    return True
#    
#    
#  set_locs = [(xs, ys)]
#  
#  while set_locs:
#    new_locs = add_2_list(set_locs, xt, yt)
#    if new_locs == True:
#      return True
#    else:
#      set_locs = remove_from_list(new_locs)
#
#  return False
#    
#    
#def add_2_list(set_locs, xt, yt):
#  new_locs = []
#  for loc in set_locs:
#    x = loc[0]
#    y = loc[1]
#    if is_in_range(x + y, y):
#      if (x + y, y) == (xt, yt) or (x, x + y) == (xt, yt):
#        return True
#      else:
#        new_locs.append((x + y, y))
#        new_locs.append((x, x + y))
#  return new_locs
#    
#    
#    
#def remove_from_list(new_locs):
#  if not new_locs == []:
#    return new_locs
#  else:
#    return False
#  
#def is_in_range(x, y):
#  return x > 1 and x < 1000 and y > 1 and y < 1000
#  
##%%
## print(can_reach(10, 20, 10, 20))  
#print(can_reach(10, 20, 80, 50))  








#%%

#def max_money(coins):
#  
#  n = len(coins)
#  
#  if n == 2:
#    return max(*coins)
#    
#  left_memo =  [-float(inf)] * n
#  right_memo = [-float(inf)] * n
#  
#  left_memo[1] = max(x[0], x[1])
#  right_memo[-2] = max(x[-1], x[2])
#  
#  
#  
#  # return max( max_money(coins[1::]) , max_money([coins[::-1]])  )
#  
#  
#print(max_money([2,3, 1]))



#%%

#def is_subset(x, S):
#  
#  n = len(x)
#  maxi = 0
#  mini = 0
#  
#  for i in range(n):
#    if x[i] > 0:
#      maxi += x[i]
#      
#    else:
#      mini += x[i]
#  
#
#  
#  if S > maxi or S < mini:
#    return False
#    
#  memo = [False] * (maxi + abs(mini) + 1)
#
#  memo[maxi] = True
#  memo[mini] = True
#  
#  memo [x[0]] = True
#  values = [x[0]]
#  
#  
#  for i in range(1, n):
#    for value in values:
#      memo[value + x[i]] = True
#  
#  return memo#[S]
#    
#print(is_subset([-1,1,2,5, -6], -7))
#  



#%%

#def num_ways(n):
#  
#  memo = [0] * (n + 1)
#  steps = [1,2,3]
#  memo[0] = 1
#  
#  for i in range(1, n + 1):
#    for step in steps:
#      if i >= step:
#        memo[i] += memo[i-step]
#        
#  return memo
#  
#  
#print(num_ways(4))  

#%%

#def LIS(x):
#
#  n = len(x)
#  if n == 1:
#    return n, x
#    
#  ind = [0]
#  memo = [0] * n
#  memo[0] = 1
#  
#  for i in range(1, n):
#    temp_ind = i
#    for j in range(i):
#      if x[i] >= x[j]:
#        memo[i] = memo[j]
#        temp_ind = j
#
#    memo[i] += 1
#    ind += [temp_ind]
#    
#  print(ind)
#  
#  seq = []
#  cur_ind = n - 1
#  while cur_ind != ind[cur_ind]:
#    seq.insert(0, x[cur_ind])
#    cur_ind = ind[cur_ind]
#    
#  seq.insert(0, x[cur_ind])
#  
#  return memo[-1], seq
#  
#print(LIS([1,2,4,-4, 1,2, 3,3])) # 5

#%%

#def fib(n):
#  
#  if n<2:
#    return n
#    
#  memo = [0] * (n + 1)
#  memo[0] = 0
#  memo[1] = 1
#  
#  if memo[n-1] == 0:
#    memo[n-1] = fib(n-1)
#  if memo[n-2] == 0:
#    memo[n-2] = fib(n-2)
#    
#  return memo[n-1]+memo[n-2]
#  
#  
#print(fib(0))
#print(fib(1))
#print(fib(2))
#print(fib(3))
#print(fib(4))
#print(fib(5))
#print(fib(6))
#print(fib(7))
#print(fib(8))
#print(fib(9))
#print(fib(10))



##%%
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
#    memo = [[0 for _ in range(n)] for __ in range(m)]  #n x m
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



















