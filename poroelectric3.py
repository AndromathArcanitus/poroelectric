# Poro-electro-elasticity Equations
#           epsilon dE/dt + sigma E - curl H - L grad p = J
#                mu dH/dt           + curl E            = 0
#      -lambda grad div u     - G div grad u  + alpha p = 0
# d/dt(c p + alpha div u) + L div E - kappa laplacian p = 0

from fenics import *
from dolfin import *
from ufl import nabla_div
import numpy as np
import matplotlib.pyplot as plt

# Definition of constants and parameters
epsilon0 = 1
sigma0 = 2*pi*pi
L0 = 0.0
mu0 = 1
lambda0 = 2
G0 = 1
alpha0 = 1
c0 = 1
kappa0 = 1/(3*pi*pi)

T = 1.0 # final time
num_steps = 10 # number of time steps
dt = T / num_steps # time step size

# Create mesh and define function space
# Load mesh
mesh = UnitCubeMesh(5, 5, 5)

# Build function space
D1 = FiniteElement("N1curl", mesh.ufl_cell(), 1)
B1 = FiniteElement("RT", mesh.ufl_cell(), 1)
V = VectorElement("Lagrange", mesh.ufl_cell(), 1)
Q = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
element = MixedElement([D1, B1, V, Q])
W = FunctionSpace(mesh, element)

# Exact solutions, boundary conditions and known functions
#E_ex = Expression(('sin(pi*x[1])*sin(pi*x[2])*exp(-t)', 'sin(pi*x[2])*sin(pi*x[0])*exp(-t)', \
#                   'sin(pi*x[0])*sin(pi*x[1])*exp(-t)'), degree = 2, t = 0)
#E_ex = Expression(('sin(pi*x[1])*sin(pi*x[2])*(1+t)', 'sin(pi*x[2])*sin(pi*x[0])*(1+t)', \
#                   'sin(pi*x[0])*sin(pi*x[1])*(1+t)'), degree = 2, t = 0)
#E_ex = Expression(('0', '0', '0'), degree = 2, t = 0)
#H_ex = Expression(('pi*sin(pi*x[0])*(cos(pi*x[1])-cos(pi*x[2]))*exp(-t)', \
#                   'pi*sin(pi*x[1])*(cos(pi*x[2])-cos(pi*x[0]))*exp(-t)', \
#                   'pi*sin(pi*x[2])*(cos(pi*x[0])-cos(pi*x[1]))*exp(-t)'), degree = 2, t = 0)
#H_ex = Expression(('-pi*sin(pi*x[0])*(cos(pi*x[1])-cos(pi*x[2]))*(t+t*t/2)', \
#                   '-pi*sin(pi*x[1])*(cos(pi*x[2])-cos(pi*x[0]))*(t+t*t/2)', \
#                   '-pi*sin(pi*x[2])*(cos(pi*x[0])-cos(pi*x[1]))*(t+t*t/2)'), degree = 2, t = 0)
#H_ex = Expression(('0', '0', '0'), degree = 2, t = 0)

J = Expression(('(10*x[0]*x[1]*(1-x[1])*x[2]*(1-x[2])-(x[1] - x[1]*x[1] + x[2] - x[2]*x[2]))*30*exp(-t)-3*(pi*cos(pi*x[0]))*sin(pi*x[1])*sin(pi*x[2])*(1+t)', \
                '(10*x[1]*x[2]*(1-x[2])*x[0]*(1-x[0])-(x[0] - x[0]*x[0] + x[2] - x[2]*x[2]))*30*exp(-t)-3*(pi*cos(pi*x[1]))*sin(pi*x[2])*sin(pi*x[0])*(1+t)', \
                '(10*x[0]*(1-x[0])*x[1]*(1-x[1])*x[2]-(x[0] - x[0]*x[0] + x[1] - x[1]*x[1]))*30*exp(-t)-3*(pi*cos(pi*x[2]))*sin(pi*x[0])*sin(pi*x[1])*(1+t)'), degree = 2, t = 0)
#J = Expression(('0', '0', '0'), degree = 2, t = 0)
u_D = Expression(('sin(pi*x[0])*sin(pi*x[1])*sin(pi*x[2])*(1+t)', '0', '0'), degree = 2, t = 0)
#p_D = Expression('0', degree = 2, t = 0)
#u_D = Expression(('0', '0', '0'), degree = 2, t = 0)
p_D = Expression('sin(pi*x[0])*sin(pi*x[1])*sin(pi*x[2])*(1+t)', degree = 2, t = 0)

f = Expression(('(5*pi*pi*sin(pi*x[0])*sin(pi*x[1])*sin(pi*x[2]) + 10*pi*cos(pi*x[0])*sin(pi*x[1])*sin(pi*x[2]))*(1+t)', \
                '(-2*pi*pi*cos(pi*x[0])*cos(pi*x[1])*sin(pi*x[2]) + 10*pi*cos(pi*x[1])*sin(pi*x[2])*sin(pi*x[0]))*(1+t)', \
                '(-2*pi*pi*cos(pi*x[0])*sin(pi*x[1])*cos(pi*x[2]) + 10*pi*cos(pi*x[2])*sin(pi*x[0])*sin(pi*x[1]))*(1+t)'), degree = 2, t = 0)
g = Expression('pi*cos(pi*x[0])*sin(pi*x[1])*sin(pi*x[2]) + \
                10*sin(pi*x[0])*(2+t)*sin(pi*x[1])*sin(pi*x[2]) + \
                9*(-(-1 + x[0])*x[0]*(-1 + x[1])*x[1] - (-1 + x[0])*x[0]*(-1 + x[2])*x[2] - (-1 + x[1])*x[1]*(-1 + x[2])*x[2])*exp(-t)', degree = 2, t = 0)
#f = Expression(('pi*cos(pi*x[0])*sin(pi*x[1])*sin(pi*x[2])*exp(-t)', \
#                'pi*cos(pi*x[1])*sin(pi*x[2])*sin(pi*x[0])*exp(-t)', \
#                'pi*cos(pi*x[2])*sin(pi*x[0])*sin(pi*x[1])*exp(-t)'), degree = 2, t = 0)
f = Expression(('0', '0', '0'), degree = 2, t = 0)
g = Expression('0', degree = 2, t = 0)
U_0 = Expression(('30*x[0]*x[1]*(1-x[1])*x[2]*(1-x[2])', '30*x[1]*x[2]*(1-x[2])*x[0]*(1-x[0])', '30*x[0]*(1-x[0])*x[1]*(1-x[1])*x[2]',\
#    'pi*sin(pi*x[0])*(cos(pi*x[1])-cos(pi*x[2]))', 'pi*sin(pi*x[1])*(cos(pi*x[2])-cos(pi*x[0]))', 'pi*sin(pi*x[2])*(cos(pi*x[0])-cos(pi*x[1]))',\
#U_0 = Expression(('0', '0', '0',\
    '30*(-1 + x[0])*x[0]*(x[1] - x[2])', '30*(-1 + x[1])*x[1]*(x[2] - x[0])', '30*(x[0] - x[1])*(-1 + x[2])*x[2]',\
    '10*sin(pi*x[0])*sin(pi*x[1])*sin(pi*x[2])', '10*sin(pi*x[0])*sin(pi*x[1])*sin(pi*x[2])', '0.5*x[1]', \
    #'0', '0', '0', \
    '100*sin(pi*x[0])*sin(pi*x[1])*sin(pi*x[2])'), degree = 2)
n = FacetNormal(mesh)
T_d = Constant((0, 0, 0))
tol = 1E-14

def boundary(x, on_boundary):
    return on_boundary

def clamped_boundary(x, on_boundary):
    return on_boundary and x[0] < tol

bcD = DirichletBC(W.sub(0), Constant((0.0, 0.0, 0.0)), DomainBoundary())
bcB = DirichletBC(W.sub(1), Constant((0.0, 0.0, 0.0)), DomainBoundary())
u_bc = Expression(('0.0', '0.0', 'x[0]'), degree = 1)
bcu = DirichletBC(W.sub(2), u_bc, DomainBoundary())
#p_bc = Expression('x[0]*x[1]', degree = 1)
bcp = DirichletBC(W.sub(3), Constant(0.0), boundary)
bc = [bcD, bcB, bcu, bcp]

# Define strain and stress
def epsilon(u):
    return 0.5*(nabla_grad(u) + nabla_grad(u).T)
    #return sym(nabla_grad(u))

def sigma(u):
    return lambda0*nabla_div(u)*Identity(3) + 2*mu0*epsilon(u)

# Define expressions used in variational forms
dt = Constant(dt)
epsilon0 = Constant(epsilon0)
sigma0 = Constant(sigma0)
L0 = Constant(L0)
mu0 = Constant(mu0)
lambda0 = Constant(lambda0)
G0 = Constant(G0)
alpha0 = Constant(alpha0)
c0 = Constant(c0)
kappa0 = Constant(kappa0)

# Define initial value
bfU_n = project(U_0, W)
E_n, H_n, bfu_n, p_n = split(bfU_n)

# Define variational problem
E, H, bfu, p = TrialFunctions(W)
D, B, bfv, q = TestFunctions(W)#

#+ inner(sigma(bfu), epsilon(bfv))*dx - alpha0*inner(p, nabla_div(bfv))*dx  - lambda0*inner(nabla_div(bfu), bfv*n)*ds\

a = epsilon0*inner(E, D)*dx +dt*sigma0*inner(E, D)*dx - dt*inner(H, curl(D))*dx - dt*L0*inner(nabla_grad(p), D)*dx\
    +  mu0*inner(H, B)*dx + dt*inner(curl(E), B)*dx\
    + lambda0*inner(nabla_div(bfu), nabla_div(bfv))*dx\
    + G0*inner(nabla_grad(bfu), nabla_grad(bfv))*dx - G0*inner(nabla_grad(bfu)*n, bfv)*ds\
    + c0*inner(p, q)*dx + alpha0*inner(nabla_div(bfu), q)*dx\
    - alpha0*inner(p, nabla_div(bfv))*dx\
    - dt*L0*inner(E, nabla_grad(q))*dx + dt*kappa0*inner(nabla_grad(p), nabla_grad(q))*dx
L = epsilon0*inner(E_n, D)*dx + mu0*inner(H_n, B)*dx \
    + inner(c0*p_n, q)*dx + alpha0*inner(nabla_div(bfu_n), q)*dx # + dt*inner(J, D)*dx + inner(f, bfv)*dx +  + dt*g
# Time-stepping#
bfU = Function(W)
t = 0
for nn in range(num_steps):

    # Update current time
    t += 0.1
    #E_ex.t = t
    #H_ex.t = t
    J.t = t
    #u_D.t = t
    #p_D.t = t
    f.t = t
    g.t = t

    # Solve variational problem
    solve(a == L, bfU, bc)

    Eh, Hh, Uh, ph = bfU.split()
    # error_L21 = errornorm(E_ex, Eh, norm_type = 'l2')
    # error_L22 = errornorm(H_ex, Hh, norm_type = 'l2')
    # error_L23 = errornorm(u_D, Uh, norm_type = 'l2')
    # error_L24 = errornorm(p_D, ph, norm_type = 'l2')
    # ENorm = sqrt(assemble(inner(E_ex, E_ex)*dx(mesh)))
    # HNorm = sqrt(assemble(inner(H_ex, H_ex)*dx(mesh)))
    # UNorm = sqrt(assemble(inner(u_D, u_D)*dx(mesh)))
    # pNorm = sqrt(assemble(inner(p_D, p_D)*dx(mesh)))
    # rel_err_21 = error_L21/ENorm
    # rel_err_22 = error_L22/HNorm
    # rel_err_23 = error_L23/UNorm
    # rel_err_24 = error_L24/pNorm
    # print('At t = ', t)
    # print('||E - E_ex|| = ', error_L21)
    # print('||H - H_ex|| = ', error_L22)
    # print('relative error ||E - E_ex||/||E|| = ', rel_err_21)
    # print('relative error ||H - H_ex||/||H|| = ', rel_err_22)
    # print('relative error ||U - U_ex||/||U|| = ', rel_err_23)
    # print('relative error ||p - p_ex||/||p|| = ', rel_err_24)

    # Update previous solution
    bfU_n.assign(bfU)
    #_un, _pn = bfU.split()
    #bfu_n.assign(_un)
    #p_n.assign(_pn)


