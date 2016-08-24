# http://opengarden.com/jobs
#
# Solve this puzzle
#
# Can you solve this puzzle? While not a requirement, we give priority consideration to candidates supplying a solution.
#
# The 2010 Census puts populations of 26 largest US metro areas at 18897109, 12828837, 9461105, 6371773, 5965343, 5946800, 5582170, 5564635, 5268860, 4552402, 4335391, 4296250, 4224851, 4192887, 3439809, 3279833, 3095313, 2812896, 2783243, 2710489, 2543482, 2356285, 2226009, 2149127, 2142508, and 2134411.
#
# Can you find a subset of these areas where a total of exactly 100,000,000 people live, assuming the census estimates are exactly right? Provide the answer and code or reasoning used.
#
# RESULT:
#
# calculated subset indices
# possible combination
# 18897109
# 12828837
# 9461105
# 6371773
# 5946800
# 5582170
# 5268860
# 4552402
# 4335391
# 4296250
# 4224851
# 3279833
# 3095313
# 2812896
# 2543482
# 2226009
# 2142508
# 2134411


import sys
from itertools import combinations, chain
#create all combinatorial subsets of the set of numbers in a list
allsubsets = lambda n: list(chain(*[combinations(range(n), ni) for ni in range(n+1)]))

area = [18897109, 12828837, 9461105, 6371773, 5965343, 5946800, 5582170, 5564635, 5268860, 4552402, 4335391, 4296250, 4224851, 4192887, 3439809, 3279833, 3095313, 2812896, 2783243, 2710489, 2543482, 2356285, 2226009, 2149127, 2142508, 2134411]

subset_indices = allsubsets(len(area))

#print subset_indices
print "calculated subset indices"

for x in subset_indices:
     index = x
     sum = 0
     #print index
     for y in range(len(index)):
         sum  = sum + area[index[y]]
     #print sum
     if(sum == 100000000):
         print "possible combination"
         for y in range(len(index)):
             print area[index[y]]
             

                 