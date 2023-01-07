




classes_id = {"klt_box_empty" : 1008, 
              "klt_box_full" : 1009,
              "rack_1" : 1200, 
              "rack_2" : 1205, 
              "rack_3" : 1210, 
              "rack_4" : 1215,
              }
num_box_per_rack = {"rack_1" : {4 : [6, 2]}, 
                    "rack_2" : {3 : [3, 3], 0 : [4, 3]}, 
                    "rack_3" : {3 : [1, 2]},
                    "rack_4" : {0 : [1, 1], 1 : [2, 2], 2 : [1, 1], 3 : [2, 2], 4 : [2, 2], 5 : [1, 1]}}
                    

rack_box = {"label" : "rack_1", "box" : [222, 333, 444, 555]}


""" Should the placeholders be added before or after training? 
    Detecting 'empty' slots is difficult
"""
def get_placeholder():
    """
    Get rack coordinates + label 
    Partition the rack in shelves
    Check for klts in shelves
    Decide Position of missing klt: from camera perspective: left, right, ....
    """         
    pass                    
                    
                    
                    