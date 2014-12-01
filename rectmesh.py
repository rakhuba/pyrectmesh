#
# RECTMESH is a class that generates 2d structured nonuniform grids in domains
# that can be represented as a combination of rectangulars.
#
# Possible domain (only simply connected so far):
#       _______        _____
#      |      |       |    |
#      |      |       |    |
#      |      |_______|    |____
#      |                       |
#      |                       |
#      |___                    |
#         |         ___        |
#         |________|  |________|
#
#
# meshx and meshy are 1d grids. Note that meshx and meshy should include points that correspond to the edges of rectagulars

import numpy as np
import scipy as sp
import scipy.sparse
import matplotlib
import matplotlib.pyplot as plt

class rectmesh:
    
    def __init__(self, meshx = None, meshy = None, nodes = None, boundary = None):
        
        self.x = meshx
        self.y = meshy
        self.mesh_size = [len(meshx), len(meshy)]
        
        self.nodes = nodes
        self.num_nodes = len(nodes)
        
        self.mask = self.make_mask()
        
        self.mask_nnz = len(np.where(self.mask>0)[0])
        
        self.dirichlet_boundary = []
        self.neumann_boundary = []
        
        self.boundary = boundary
        
        # Order of elements
        # Call it when your boundary conditions are ready
        self.order = None
        self.num_inside = None
        self.num_neumann = None
        self.num_dirichlet = None
        self.dirichlet_values = None
        return
    
    def make_mask(self): # winding points way
        mask = np.ones(self.mesh_size) #sp.sparse.coo_matrix(self.mesh_size)
        droptol = 1e-12
        
        #cdef int i
        #cdef int j
        #cdef int n
        for i in xrange(self.mesh_size[0]):
            for j in xrange(self.mesh_size[1]):
                ang = 0.
                vec = np.zeros((self.num_nodes+1, 2))
                #cdef double[:, :] cview = vec
                for n in xrange(self.num_nodes):
                    vec[n] = np.array([self.x[self.nodes[n,0]] - self.x[i], self.y[self.nodes[n,1]] - self.y[j]])
                #vec[self.num_nodes] = np.array([self.x[self.nodes[0,0]] - self.x[i], self.y[self.nodes[0,1]] - self.y[j]])
                
                for n in xrange(self.num_nodes - 1):
                    scalar = vec[n,0]*vec[n+1, 0] + vec[n, 1]*vec[n+1, 1]
                    norm1 = np.sqrt(vec[n, 0]**2 + vec[n, 1]**2)
                    norm2 = np.sqrt(vec[n+1, 0]**2 + vec[n+1, 1]**2)
                    sign_det = np.sign(vec[n, 0]*vec[n+1, 1] - vec[n, 1]*vec[n+1, 0])
                    
                    if norm1 == 0. or norm2 == 0.:
                       ang = 1.
                       break
                    
                    ang += np.arccos(scalar / (norm1 * norm2)) * sign_det
                if abs(ang) < droptol:
                    mask[i,j] = 0.
    
        return mask

    def add_dirichlet_boundary(self, ind1, ind2, values): #ind1 or ind2 must be of size 1 !
        length = max(len(ind1), len(ind2))
        self.mask[ind1, ind2] = 2*np.ones(length)
        self.dirichlet_boundary.append([ind1, ind2, values])
        return
    
    def add_neumann_boundary(self, ind1, ind2, values):
        length = max(len(ind1), len(ind2))
        self.mask[ind1, ind2] = 3*np.ones(length)
        self.neumann_boundary.append([ind1, ind2, values])
        return
    
    def add_neumann_neumann_boundary(self, ind1, ind2, values):
        length = max(len(ind1), len(ind2))
        self.mask[ind1, ind2] = 4*np.ones(length)
        self.neumann_boundary.append([ind1, ind2, values])
        return
    
    def create_boundary(self):
        
        for i in xrange(len(self.boundary)):
            
            ind10 = self.nodes[i, 0]
            ind11 = self.nodes[i + 1, 0]
                
            ind20 = self.nodes[i, 1]
            ind21 = self.nodes[i + 1, 1]
            
            
            if ind10 == ind11:
                
                ind1 = [ind10]
                
                if ind21 > ind20:
                    ind2 = range(ind20, ind21 + 1)
        
                elif ind21 < ind20:
                    ind2 = list(reversed(range(ind21, ind20 + 1)))

                else:
                    ind2 = [ind21]
        
            elif ind20 == ind21:
            
                ind2 = [ind20]
            
                if ind11 > ind10:
                    ind1 = range(ind10, ind11 + 1)
            
                elif ind11 < ind10:
                    ind1 = list(reversed(range(ind11, ind10 + 1)))
                        
                else:
                    ind1 = [ind11]

            else:
                raise Exception('Check nodes')
            
        
            if self.boundary[i][0] == 'D':
                    
                if self.boundary[i - 1][0] == 'D':
                    
                    if ind10 == ind11:
                        self.add_dirichlet_boundary(ind1, ind2[1:], self.boundary[i][1])
                    else:
                        self.add_dirichlet_boundary(ind1[1:], ind2, self.boundary[i][1])
    
                if self.boundary[i - 1][0] == 'N':
                    
                    self.add_dirichlet_boundary(ind1, ind2, self.boundary[i][1])
    
    
            elif self.boundary[i][0] == 'N':
    
                if ind10 == ind11:
                    self.add_neumann_boundary(ind1, ind2[1: -1], self.boundary[i][1])
                else:
                    self.add_neumann_boundary(ind1[1: -1], ind2, self.boundary[i][1])
                
                        

            if self.boundary[i][0] == self.boundary[i-1][0] and self.boundary[i][0] == 'N':
                self.mask[self.nodes[i][0], self.nodes[i][1]] = 4 # Neumann-Neumann point
                self.add_neumann_neumann_boundary([ind1[0]], [ind2[0]], self.boundary[i][1])
        return
    
    def create_order(self):
        
        order = {} #np.zeros((self.mask_nnz, 2))
        
        
        el_num = 0
        for i in xrange(self.mesh_size[0]):
            for j in xrange(self.mesh_size[1]):
                if self.mask[i,j] == 1:
                    order[(i,j)] = el_num
                    el_num += 1
        self.num_inside = el_num
        
        for i in xrange(len(self.neumann_boundary)):
            ind1 = self.neumann_boundary[i][0]
            ind2 = self.neumann_boundary[i][1]
            
            for j in xrange(len(ind1)):
                for k in xrange(len(ind2)):
                    order[(ind1[j], ind2[k])] = el_num
                    el_num += 1
        self.num_neumann = el_num - self.num_inside
        
        self.dirichlet_values = {}
        for i in xrange(len(self.dirichlet_boundary)):
            ind1 = self.dirichlet_boundary[i][0]
            ind2 = self.dirichlet_boundary[i][1]
            value = self.dirichlet_boundary[i][2]
            
            for j in xrange(len(ind1)):
                for k in xrange(len(ind2)):
                    self.dirichlet_values[el_num] = value
                    order[(ind1[j], ind2[k])] = el_num
                    el_num
                    el_num += 1
        self.num_dirichlet = el_num - (self.num_inside + self.num_neumann)
    
        self.order = order
        
        if el_num <> self.mask_nnz:
            raise Exception('Check your boundary condition sizes')
        
        return
    
    
    
    def plot(self):
        lines = []
        for i in xrange(self.mesh_size[0]-1):
            for j in xrange(self.mesh_size[1]-1):
                if self.mask[i,j]>0:
                    if self.mask[i+1,j] <> 0.:
                        lines.append(plt.Line2D((self.x[i], self.x[i+1]), (self.y[j], self.y[j]), color='k'))
                    if self.mask[i,j+1] <> 0.:
                        lines.append(plt.Line2D((self.x[i], self.x[i]), (self.y[j], self.y[j+1]), color='k'))
        
        i = self.mesh_size[0] - 1
        for j in xrange(self.mesh_size[1]-1):
            if self.mask[i,j]>0:
                if self.mask[i,j+1] <> 0.:
                    lines.append(plt.Line2D((self.x[i], self.x[i]), (self.y[j], self.y[j+1]), color='k'))
    
        j = self.mesh_size[1] - 1
        for i in xrange(self.mesh_size[0]-1):
            if self.mask[i,j]>0:
                if self.mask[i+1,j] <> 0.:
                    lines.append(plt.Line2D((self.x[i], self.x[i+1]), (self.y[j], self.y[j]), color='k'))
    
        fig = plt.gcf()
        for i in xrange(len(lines)):
            fig.gca().add_artist(lines[i])


