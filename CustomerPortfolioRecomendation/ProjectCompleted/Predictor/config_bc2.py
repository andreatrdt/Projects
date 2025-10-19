# For single-label classification we implement our metric, based on a penalty term that takes into account the severity of the error type
cost_map_single = {
    (0, 0): 0,
    (0, 1): 5,  # False Positive
    (1, 0): 8,  # False Negative
    (1, 1): 0
}


cost_dict = {
    # Actual (0,0)
    ((0,0), (0,0)): 0,
    ((0,0), (1,1)): 10,
    ((0,0), (1,0)): 5,
    ((0,0), (0,1)): 7,
    
    # Actual (1,1)
    ((1,1), (1,1)): 0,
    ((1,1), (1,0)): 2,
    ((1,1), (0,1)): 2,
    ((1,1), (0,0)): 10,
    
    # Actual (1,0)
    ((1,0), (1,0)): 0,
    ((1,0), (1,1)): 6,
    ((1,0), (0,1)): 8,
    ((1,0), (0,0)): 10,
    
    # Actual (0,1)
    ((0,1), (0,1)): 0,
    ((0,1), (1,0)): 7,
    ((0,1), (1,1)): 6,
    ((0,1), (0,0)): 10,
}