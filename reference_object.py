class ref_object():

    def __init__(self, width, length, unit_length):
        """
            :param unit_length: 
        """
        self.width = width
        self.length = length
        self.unit_length = unit_length 

def get_ref_object_9x11():
    obj = ref_object(7, 9, 0.78740157480314965088)         
    return obj

