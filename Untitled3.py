#!/usr/bin/env python
# coding: utf-8

# In[1]:


# MCP neuron function for OR gate
def mcp_neuron_or(x1, x2):
    w1, w2 = 1, 1      # weights
    theta = 1          # threshold
    weighted_sum = w1 * x1 + w2 * x2
    return 1 if weighted_sum >= theta else 0

# Test all input combinations for OR gate
inputs = [(0,0), (0,1), (1,0), (1,1)]

print("OR Gate using MCP neuron:")
for x1, x2 in inputs:
    output = mcp_neuron_or(x1, x2)
    print(f"Input: ({x1}, {x2}) -> Output: {output}")


# In[2]:


# MCP neuron function for AND gate
def mcp_neuron_and(x1, x2):
    w1, w2 = 1, 1      # weights
    theta = 2          # threshold
    weighted_sum = w1 * x1 + w2 * x2
    return 1 if weighted_sum >= theta else 0

# Test all input combinations for AND gate
inputs = [(0,0), (0,1), (1,0), (1,1)]

print("AND Gate using MCP neuron:")
for x1, x2 in inputs:
    output = mcp_neuron_and(x1, x2)
    print(f"Input: ({x1}, {x2}) -> Output: {output}")


# In[ ]:




