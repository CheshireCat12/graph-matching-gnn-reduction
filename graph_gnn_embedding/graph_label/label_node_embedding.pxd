from graph_pkg.graph.label.label_base cimport LabelBase
cimport numpy as cnp

cdef class LabelNodeEmbedding(LabelBase):
    cdef:
        readonly cnp.ndarray vector

    cpdef tuple get_attributes(self)
