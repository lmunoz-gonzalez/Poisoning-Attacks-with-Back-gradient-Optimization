function cost = LRcost(s,d)
cost = - (d.*log(s) + (1-d).*log(1 - s));
end