import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy as sp
from matplotlib.text import Annotation
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.patches import FancyArrowPatch

np.set_printoptions(linewidth=1000, threshold=10000, precision=5)

from forbiddenfruit import curse

curse(np.ndarray,'H',property(fget=lambda A: A.conj().T))

def dag(X):
    return X.H

def ket(i, dim, **args ):
    tmp = np.zeros([dim,1],**args)
    tmp[i,0]=1
    return tmp

def bra(i, dim, **args):
    tmp = np.zeros([1,dim],**args)
    tmp[0,i]=1
    return tmp

def X(**args):
    return np.array([[0,1],[1,0]], **args)

def Y(**args):
    return np.array([[0,-1j],[1j,0]], **args)

def Z(**args):
    return np.array([[1,0],[0,-1]], **args)

def P(**args):
    return np.array([[0,1],[0,0]], **args)

def M(**args):
    return np.array([[0,0],[1,0]], **args)

def ketbra(i,j,dim, **args):
    tmp = np.zeros([dim,dim], **args)
    tmp[i,j]=1
    return tmp


def plot_matrix(A, name='', figsize=(7,4), dims=[], ret = False):
    '''
        Draws the real and immaginary part of a given matrix.
        
        dims is a list of dimensions used to plot the blocks of the Wedderburn decomposition.
    '''
    vmin = np.min([np.real(A), np.imag(A)])
    vmax = np.max([np.real(A), np.imag(A)])
    
    fig, axs = plt.subplots(1,2,figsize=figsize)
    f1 = axs[0].matshow(np.real(A), vmin=vmin, vmax=vmax)
    axs[0].title.set_text(r'$\Re('+name+r')$')
    f2 = axs[1].matshow(np.imag(A), vmin=vmin, vmax=vmax)
    axs[1].title.set_text(r'$\Im('+name+r')$')
 
    n = 0
    for s,t in dims:
        if s==0 or t==0:
            n += s+t
        else:
            n += s*t
    
    zero_coord = -0.5
    for s,t in dims:
        if s==0 or t==0:
            d = s+t
        else:
            d = s*t
        axs[0].plot([zero_coord,zero_coord],[-0.5, n-0.5], 'k')
        axs[0].plot([-0.5, n-0.5],[zero_coord,zero_coord], 'k')
        axs[1].plot([zero_coord,zero_coord],[-0.5, n-0.5], 'k')
        axs[1].plot([-0.5, n-0.5],[zero_coord,zero_coord], 'k')
        zero_coord += d

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.01, 0.7])
    plt.colorbar(f1, cax=cbar_ax)
    if len(name) > 0:
        fig.suptitle(r'$ '+name+r'$')
    plt.show()
    if ret:
        return fig
    else:
        return
    
def multi_kron(op_list):
    if len(op_list)==1:
        return op_list[0]
    else:
        tmp = np.kron(op_list[-2], op_list[-1])
        for v in reversed(op_list[0:-2]):
            tmp = np.kron(v,tmp)
        return tmp
    
def u_ith(U, i, N):
    if i>= N:
        print('SIZE ERROR')
        return
    if N==1:
        return U
    if i == 0:
        op_list = [U]
        for k in range(N-1):
            op_list.append(np.eye(2))
    elif i == N-1:
        op_list = []
        for k in range(N-1):
            op_list.append(np.eye(2))
        op_list.append(U)
    else:
        op_list = []
        for k in range(i):
            op_list.append(np.eye(2))
        op_list.append(U)
        for k in range(i, N-1):
            op_list.append(np.eye(2))
    return multi_kron(op_list)

def partial_trace(X,n_1,n_2):
    tmp = np.zeros((n_1,n_1), dtype=np.complex128)
    for i in range(n_1):
        for j in range(n_1):
            for k in range(n_2):
                tmp[i,j] = tmp[i,j] + np.kron(bra(i,n_1),bra(k,n_2))@X@np.kron(ket(j,n_1),ket(k,n_2))
    return tmp


class Annotation3D(Annotation):

    def __init__(self, text, xyz, *args, **kwargs):
        super().__init__(text, xy=(0, 0), *args, **kwargs)
        self._xyz = xyz

    def draw(self, renderer):
        x2, y2, z2 = proj_transform(*self._xyz, self.axes.M)
        self.xy = (x2, y2)
        super().draw(renderer)
        
def _annotate3D(ax, text, xyz, *args, **kwargs):
    '''Add anotation `text` to an `Axes3d` instance.'''

    annotation = Annotation3D(text, xyz, *args, **kwargs)
    ax.add_artist(annotation)

setattr(Axes3D, 'annotate3D', _annotate3D)

class Arrow3D(FancyArrowPatch):

    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)
        
    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs) 
    
def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    '''Add an 3d arrow to an `Axes3D` instance.'''

    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)


setattr(Axes3D, 'arrow3D', _arrow3D)
        
        
def plot_bloch_shpere():
    fig = plt.figure(figsize=(21,9))
    ax = fig.add_subplot(projection='3d')
    ax.set_axis_off()

    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 10)

    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    # use scipy to interpolate
    xdata = sp.ndimage.zoom(x, 3)
    ydata = sp.ndimage.zoom(y, 3)
    zdata = sp.ndimage.zoom(z, 3)

    ax.plot_surface(xdata, ydata, zdata, rstride=3, cstride=3, color='k', alpha=0.05, edgecolor='k')
    
    ax.annotate3D(r'$x$',(1,0,0), xytext=(-10, -10), textcoords='offset points')
    ax.annotate3D(r'$y$',(0,1,0), xytext=(-10, -10), textcoords='offset points')
    ax.annotate3D(r'$z$',(0,0,1), xytext=(0, 10), textcoords='offset points')
    
    ax.arrow3D(-1,0,0, 2,0,0, mutation_scale=10, arrowstyle="->",color='k', alpha=0.5)
    ax.arrow3D( 0,-1,0, 0,2,0, mutation_scale=10, arrowstyle="->",color='k', alpha=0.5)
    ax.arrow3D( 0,0,-1, 0,0,2, mutation_scale=10, arrowstyle="->",color='k', alpha=0.5)
    
    ax.view_init(30,30)
    
    return fig, ax

def bloch_sphere_components(rho_trajectory):
    xs = []
    ys = []
    zs = []
    for rho in rho_trajectory:
        xs.append(np.trace(X()@rho))
        ys.append(np.trace(Y()@rho))
        zs.append(np.trace(Z()@rho))
        
    xs = np.real(np.array(xs))
    ys = np.real(np.array(ys))
    zs = np.real(np.array(zs))
    return xs, ys, zs

def add_arrow_to_line2D(
    axes, x,y,z,line, arrow_locs=[0.2, 0.4, 0.6, 0.8],
    arrowstyle='-|>', arrowsize=1, transform=None):
    """
    Add arrows to a matplotlib.lines.Line2D at selected locations.

    Parameters:
    -----------
    axes: 
    line: Line2D object as returned by plot command
    arrow_locs: list of locations where to insert arrows, % of total length
    arrowstyle: style of the arrow
    arrowsize: size of the arrow
    transform: a matplotlib transform instance, default to data coordinates

    Returns:
    --------
    arrows: list of arrows
    """

    arrow_kw = {
        "arrowstyle": arrowstyle,
        "mutation_scale": 10 * arrowsize,
    }

    color = line.get_color()
    use_multicolor_lines = isinstance(color, np.ndarray)
    if use_multicolor_lines:
        raise NotImplementedError("multicolor lines not supported")
    else:
        arrow_kw['color'] = color

    linewidth = line.get_linewidth()
    if isinstance(linewidth, np.ndarray):
        raise NotImplementedError("multiwidth lines not supported")
    else:
        arrow_kw['linewidth'] = linewidth

    if transform is None:
        transform = axes.transData

    arrows = []
    for loc in arrow_locs:
        s = np.cumsum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))
        n = np.searchsorted(s, s[-1] * loc)
        arrow_tail = (x[n], y[n], z[n])
        arrow_head = (np.mean(x[n:n + 2]), np.mean(y[n:n + 2]), np.mean(z[n:n + 2]))
        p = Arrow3D( x[n], y[n], z[n], x[n]-np.mean(x[n:n + 2]), y[n]-np.mean(y[n:n + 2]), z[n]-np.mean(z[n:n + 2]),
            transform=transform,
            **arrow_kw)
        axes.add_patch(p)
        arrows.append(p)
    return arrows

def draw_trajectory(xs, ys, zs, ax, label='', **args):
    line, = ax.plot(xs, ys, zs, '-', label=label, **args)
    ax.scatter(xs[0], ys[0], zs[0], 'o', facecolors='none', edgecolors=line.get_color(), **args)
    add_arrow_to_line2D(ax, xs, ys, zs, line, arrowstyle='<-')


def vec(X, base=None):
    """
    Vectorizes a matrix in a given basis. If basis is None that it is considered to be the standard matrix basis. 

    Args:
        X: A NumPy array representing the matrix.
        
        base (optional): A list of NumPy arrays representing the basis.
                        If basis is None then the matrix is vectorized in the standard basis, 
                        i.e. by stacking a column over the other starting from the first and proceding 
                        to the second under that and so on and so forth.

    Returns:
        A NumPy array representing the vectorized matrix.
    """
    if base is None:
        return X.T.reshape([X.shape[0]*X.shape[1],1]).astype(np.complex128)
    else: 
        v = np.zeros([len(base),1], dtype=np.complex128)
        for i,b in enumerate(base):
            v[i,0] = np.trace(dag(b)@X)
        return v

def matrix_from_list(operator_list, base=None):
    ''' 
    Given a list of operator this function returns a matrix whose 
    columns are the vectorizations of each operators in the list.
    
    Args: 
        operator_list: set of operators to vectorize
        base (optional): base for the vectorization of the operators.
                        If base = None then the standard base is considered 
                        
    Returns:
        A matrix M with a number of colums equal to the number of elements in the operator list and a number of rows equal to the dimension of the given base. 
    '''
    M = vec(operator_list[0], base=base)
    for o in operator_list[1:]:
        M = np.concatenate((M,vec(o, base=base)), axis=1)
    return M

def list_from_matrix(M, n=None, base=None):
    '''
        Given a matrix whose columns are the vector representation of matrices, this method returns a list of the corresponding unvectorized vectors.
        This method is basically the opposite of the method matrix_from_list. 
    '''
    lst = []
    for i in range(M.shape[1]):
        lst.append(unvec(M[:,i], n=n, base=base))
    return lst

def unvec(v, n=None, base=None):
    '''
    Given a vector, this function returns its matrix form in the given base.
    If base = None, the standard base is used.
    
    Args:
        v: a complex vector
        n: the number of rows of the matrix returned by the method
        base (optional): thr considered base.
        
    Returns: 
        An n times n complex matrix. 
    '''
    if n is None:
        n = int(np.sqrt(v.shape[0]))
    if base is None:
        return v.reshape([n,n]).T
    if len(base) != v.shape[0]:
        print('Attention! length of the base and of the vector are different')
    X = np.zeros([n,n], dtype=np.complex128)
    for i,b in enumerate(base):
        X += v[i]*b
    return X


def superoperator_to_matrix(superoperator, in_base, out_base):
    '''
    Given a superoperator, returns its matrix representation in a given base.
    '''
    
    A = np.zeros([len(out_base), len(in_base)], dtype=np.complex128)
    for i, Ei in enumerate(out_base):
        for j, Ej in enumerate(in_base):
            A[i,j] = np.trace(superoperator(Ej)@Ei.H)
    return A

def unitary_sop_to_matrix(U,base=None):
    '''
    Given a unitary matrix, it returns the matrix representation of the super operator U X U.H in a given base.
    If base = None, then the standard base is used.
    '''
    X = np.kron(np.conjugate(U),U)
    if base is None:
        return X
    else:
        T = matrix_from_list(base)
        T_ = np.linalg.pinv(T)
        return T_@X@T


def Lindblad_to_matrix(H,noise_operators, base = None):
    '''
    Given an Hamiltonian H (complex matrix) and a list of noise operators it returns the matrix representation of the Lindblad generator in a given base.
    If base = None, then the standard base is used.
    '''
    
    n = H.shape[0]
    L = -1j*(np.kron(np.eye(n),H)-np.kron(H.T,np.eye(n)))
    for K in noise_operators:
        L += np.kron(np.conjugate(K),K) -0.5*(np.kron(np.eye(n),K.H@K)+np.kron(K.T@np.conjugate(K),np.eye(n)))
    if base is None:
        return L
    else:
        T = matrix_from_list(base)
        T_ = np.linalg.pinv(T)
        return T_@L@T
    
def GellMann_base(n, orthonormal=False):
    '''GellMann Returns all the GellMann matrices of rank n
         This function returns a list of n**2
        of (nXn) Generalized Gell-Mann matrices 
        
        ATTENTION: To form an othogonal basis you need to add the 
        normalized identity, for this there is the label base=True
        The returned base is not normalized. 
        '''
    
    GM = []
    GM.append(np.eye(n, dtype=np.complex128)   ) #/ np.sqrt(n)

    for j in range(n):
        for k in range(j + 1, n):
            GM.append((ketbra(k, j, n) + ketbra(j, k, n)) ) #/ np.sqrt(2)
            GM.append((-1j * (ketbra(j, k, n) - ketbra(k, j, n))) ) #/ np.sqrt(2)

    for l in range(1, n):
        q = np.zeros([n, n], dtype=np.complex128)
        for j in range(l):
            q += ketbra(j, j, n)
        GM.append((np.sqrt(2 / (l * (l + 1))) * (q - l * ketbra(l, l, n))) ) #/ np.sqrt(2)
        
    if orthonormal:
        tmp = []
        for b in GM:
            tmp.append(b/np.sqrt(np.trace(b.H@b)))
        GM = tmp
    
    return GM

def standard_base(n,m):
    ''' 
        Returns the standard base of the space of n x m complex matrices
    '''
    basis = []
    for j in range(m):
        for i in range(n):
            basis.append(ket(i,n)@bra(j,m))
    return basis