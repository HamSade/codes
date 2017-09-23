#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 19:42:25 2017

@author: hsadeghi
"""


def longest_subseq(A, B):
    
    m = len(A)
    n = len(B)
    
    memo = [[0] * n] * m  #n x m
    seq = []
    
    if A[0] == B[0]:
        memo[0][0] = 1
        seq.append( A[0] )
        
    ###First col
    for row in range(1, m):
        if A[row] == B[0]:
            memo[row][0] = 1

    for i in range(len(memo)):
        print(memo[i][:])
            
    # First row    
    for col in range(1, n):
        if A[0] == B[col]:
            memo[col][0] = 1
            
#    for i in range(len(memo)):
#        print(memo[i][:])
            
#    for i in range(len(memo)):
#        print(memo[i])
            
#    for row in range(1, m):
#        for col in range(1, n):
#            if A[row] == B[col]:
#                memo[row][col] = memo[row - 1][col - 1] + 1
#                seq.append( A[row] )
#                print(row, col)
#            else:
#                memo[row][col] = max( memo[row - 1][col], memo[row][col - 1])
    
    # finding the largest sequence

    return memo, seq
            
    
M, S =  longest_subseq('baba', 'abab') 

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



















