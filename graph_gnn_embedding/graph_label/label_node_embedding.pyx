import random
import numpy as np
cimport numpy as cnp
cimport cython

# @cython.auto_pickle(True)
cdef class LabelNodeEmbedding(LabelBase):

    # def __cinit__(self, cnp.ndarray vector):
    #     self.vector = vector

    def __init__(self, cnp.ndarray vector):
        self.vector = vector

    cpdef tuple get_attributes(self):
        return (self.vector, )

    def sigma_attributes(self):
        return f'Element: {self.vector}'

    def sigma_position(self):
        return [random.uniform(0, 1) for _ in range(2)]

    # def __setstate__(self, state):
    #     self.vector = state
    #
    # def __getstate__(self):
    #     return self.vector.base

    # def __reduce__(self):
    #     cdef dict d = dict()
    #
    #     d['vec'] = cnp.asarray(self.vector)
    #
    #     return (rebuild, (d,))

    # def __reduce__(self):
    #     return self.__class__, (self.vector.base, )
    # def __reduce__(self):
    #     cdef dict d = dict()
    #     d['vec'] = self.vector.base # #np.asarray(self.vector)
    #     return self.__class__, (d['vec'], ), d
    #
    # def __setstate__(self, state):
    #     self.vector = state['vec']
#
# def rebuild(data):
#     cdef LabelNodeEmbedding lbl = LabelNodeEmbedding.__new__(LabelNodeEmbedding,
#                                                              data['vec'])
#
#     return lbl