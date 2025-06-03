#! /usr/bin/env python3
from system_of_bps import *
from random import Random
import matplotlib.pyplot as plt
from scipy.special import lambertw

#plt.rcParams.update({
#    "text.usetex": True,        
#    "font.family": "serif",     
#    "font.size": 12,
#    'axes.labelsize': 12,
#    'axes.titlesize': 12,
#    'xtick.labelsize': 9,
#    'ytick.labelsize': 9
#}) #Paper style

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amssymb}',
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'lines.linewidth': 1.5,
    'figure.dpi': 300
})

    
def AlternateApproximateSolution(rng, x, maxNtrees, t, type_dict):
    # Takes:
    # - x:          an m-dimensional vector taking values in [0, \infty)^m
    # - maxNtrees:  the number of trees up to which we approximate the true solution
    # - t_cutoff:   cutoff time
    # - type_dict:  defaultdict containing all types with their coefficients

    # Returns:
    # - a list of time and associated u(t, x) approximation

    # Procedure outline:
    #   - Convert type_dict to a modified coefficient dict associated to x
    #   - For each component generate Ntree trees, for each time point, average the trees so far
    #   - Return a sequence of vectors indexed by time

    alpha_cutoff = 1000000
    m = len(x)

    #convert coefficients
    l1_types = GetLevelTypes(m, 1, type_dict)                   #Obtain level 1 types
    l2_types = GetLevelTypes(m, 2, type_dict)                   #Obtain level 2 types
    sol_type_dict = TypesAtX(x, l1_types, l2_types, type_dict)

    ### FOR SOME EXTRA EFFICIENCY, WE COULD REMOVE ZERO COEFFICIENTS IF TRULY NECESSARY
    u_appr = []
    for k in range(1, m + 1):
        u_k_appr = []
        tot = 0.0
        for n in range(1, maxNtrees + 1):
            tot += GenerateTree(rng, k, t, alpha_cutoff, l1_types, l2_types, sol_type_dict)[-1][1]
            u_k_appr.append(tot/n)
        u_appr.append(u_k_appr)
    return u_appr

def test_case_1():
    rng = Random()
    rng.seed(1)

    N_trees = 10000
    t_cutoff = 0.4
    # Test case 1 with the following initial condition and nonlinearity:
    # - u_1(0, x) = 0.5 e^{-x_1} + 0.5 e^{-x_2}
    # - u_2(0, x) = -0.5 e^{-x_1} + 1.5 e^{- 2 x_2}
    # - F_1(s) = s_1 + s_2
    # - F_2(s) = s_1 * s_2
    # x = [1, 1]
    t1_td = defaultdict(float)
    # type vectors are encoded as tuples (level, component, mass_1, ..., mass_m)
    t1_td[(1, 1, 1, 0)] = 0.5
    t1_td[(1, 1, 0, 1)] = 0.5
    t1_td[(1, 2, 1, 0)] = -0.5
    t1_td[(1, 2, 0, 2)] = 1.5
    t1_td[(2, 1, 1, 0)] = 1.0
    t1_td[(2, 1, 0, 1)] = 1.0
    t1_td[(2, 2, 1, 1)] = 1.0
    # Need to add implementation for the above conditions
    test_1_x = (1, 1)
    test_1_u = ApproximateSolution(rng, test_1_x, N_trees, t_cutoff, t1_td)
    #print(f"The resulting value at x = (1, 1), t = 0.1 is: {test_1_u(t)}")
    t_range = np.linspace(0, t_cutoff, 100)

    u1 = [test_1_u(t)[0] for t in t_range]
    u2 = [test_1_u(t)[1] for t in t_range]

    print('c')

    fig, axs = plt.subplots(1, 2, figsize=(3.8, 2.2), constrained_layout=True)

    #plot 1
    axs[0].plot(t_range, u1)
    axs[0].set_xlabel(r"t")
    axs[0].set_title(r"$\tilde{u}_1(t, \mathbf{x})$")

    #plot 2
    axs[1].plot(t_range, u2)
    axs[1].set_xlabel(r"t")
    axs[1].set_title(r"$\tilde{u}_2(t, \mathbf{x})$")

    fig.suptitle(r"Approximation of $t \mapsto \tilde{\mathbf{u}}(t, \mathbf{x})$")

    plt.savefig("2D_nonlinear_time_simulation.pdf", bbox_inches = "tight", transparent = True)
    plt.show()

def test_case_2():
    #Test is based on a 2D wave equation
    rng = Random()
    rng.seed(2)

    N_trees = 100000
    t = 0.3

    # Test case 1 with the following initial condition and nonlinearity:
    # - u_1(0, x) = 0.5 e^{-x_1} + 0.5 e^{-x_2}
    # - u_2(0, x) = -0.5 e^{-x_1} + 1.5 e^{- 2 x_2}
    # - F_1(s) = 1
    # - F_2(s) = 1
    # x = [1, 1]

    t1_td = defaultdict(float)
    # type vectors are encoded as tuples (level, component, mass_1, ..., mass_m)
    t1_td[(1, 1, 1, 0)] = 0.5
    t1_td[(1, 1, 0, 1)] = 0.5
    t1_td[(1, 2, 1, 0)] = -0.5
    t1_td[(1, 2, 0, 2)] = 1.5
    t1_td[(2, 1, 0, 0)] = 1.0
    t1_td[(2, 2, 0, 0)] = 1.0
    # Need to add implementation for the above conditions
    test_1_x = (1, 1)
    test_1_u = AlternateApproximateSolution(rng, test_1_x, N_trees, t, t1_td)
    #print(f"The resulting value at x = (1, 1), t = 0.1 is: {test_1_u(t)}")

    def real_u(t, x):
        x1 = x[0]
        x2 = x[1]
        return [0.5 *exp(-x1 + t) + 0.5 * exp(-x2 + t), -0.5 * exp(-x1 + t) + 1.5*exp(-2*x2 + 2*t)]

    u1 = test_1_u[0]
    u2 = test_1_u[1]
    abs_err = []
    rel_err = []
    for i in range(N_trees):
        real_u_val = real_u(t, test_1_x)
        diff1 = abs(real_u_val[0] - u1[i])
        diff2 = abs(real_u_val[1] - u2[i])
        abs_err.append(diff1 + diff2)
        rel_err.append(diff1/real_u_val[0] + diff2/real_u_val[1])

    # First plot shows how the value stabilizes as the number of trees goes up.
    fig, axs = plt.subplots(1, 2, figsize=(3.8, 2.2), constrained_layout=True)

    # plot 1
    axs[0].plot(range(N_trees), u1, linewidth = 1.2)
    axs[0].set_xscale('log')
    axs[0].set_xlabel(r'$N_{\mathrm{trees}}$')
    axs[0].set_title(r"$\tilde{u}_1(t, \mathbf{x})$")

    # plot 2
    axs[1].plot(range(N_trees), u2, label = r'$u_2(t, x)$', linewidth = 1.2)
    axs[1].set_xscale('log')
    axs[1].set_xlabel(r'$N_{\mathrm{trees}}$')
    axs[1].set_title(r"$\tilde{u}_2(t, \mathbf{x})$")

    fig.suptitle(r"2D wave equation convergence of $\tilde{\mathbf{u}}§(t, \mathbf{x})$ at $\mathbf{x} = (1.0, 1.0)$, $t = 0.3$",
    fontsize=9
)
    plt.savefig("2DWaveEquationConvergence.pdf", dpi=300, bbox_inches = "tight", pad_inches = 0.02, transparent = True, format = 'pdf')

    # Second plot show how the relative error changes as the number of trees goes up.

    fig2, ax2 = plt.subplots(figsize=(3.4, 2.0), constrained_layout=True)

    ax2.plot(range(N_trees), rel_err, linewidth = 1.2)
    ax2.set_yscale('log')
    ax2.set_xscale('log')
    ax2.set_xlabel(r'$N_{\mathrm{trees}}$')
    ax2.set_title(r"Error of 2D wave equation" "\n" r"at $\mathbf{x} = (1.0, 1.0)$, $t = 0.3$", fontsize=9)
    Ntree_range = np.logspace(1, 5, num = 5, dtype = int)
    ax2.plot(Ntree_range, [3.0/float(np.sqrt(N)) for N in Ntree_range], '--', linewidth=1, label=r'$1/\sqrt{N}$')
    ax2.legend()
    plt.savefig("2DWaveEquationRelError.pdf", bbox_inches = "tight", transparent = True)
    plt.tight_layout()
    plt.show()

def test_1D_inviscid_burgers():
    rng = Random()
    Ntrees = 10000
    t_choice = 0.90

    td = defaultdict(float)

    #u(0, x) = e^{-x}
    #F(s) = s - 1
    td[(1, 1, 1)] = 0.5
    td[(2, 1, 1)] = 1.0
    td[(2, 1, 0)] = -1.0

    x_range = np.linspace(0, 2, 20)
    u = []
    u_expl = []
    for x in x_range:
        u_apprx = ApproximateSolution(rng, [x], Ntrees, t_choice, td)
        u.append(u_apprx(t_choice))
        u_expl.append(-1.0/t_choice * lambertw(-0.5*t_choice * exp(-x-t_choice)))
    
    fig, ax = plt.subplots()

    ax.plot(x_range, u)
    ax.plot(x_range, u_expl)

    ax.set_xlabel('x')
    ax.set_ylabel('')
    ax.set_title('Plot of u_t = -u u_x + u_x, u(0, x) = exp(-x)')
    plt.show()

def test_1D_wave_equation():
    rng = Random()
    rng.seed(2)

    N_trees = 10000
    t_cutoff = 0.9
    t_choice = 0.8

    td = defaultdict(float)

    #u(0, x) = 0.5e^{-x} - 0.5 e^{-2x}
    #F(s) = 1
    td[(1, 1, 1)] = 0.5
    td[(1, 1, 2)] = -0.5
    td[(2, 1, 0)] = 1.0

    x_range = np.linspace(0, 5, 20)
    u = []
    u_init = []
    for x in x_range:
        u_apprx = ApproximateSolution(rng, [x], N_trees, t_cutoff, td)
        u.append(u_apprx(t_choice))
        u_init.append(0.5*exp(-1.0*(x-t_choice)) - 0.5*exp(-2.0*(x - t_choice)))
    
    fig, ax = plt.subplots()

    ax.plot(x_range, u)
    ax.plot(x_range, u_init)

    ax.set_xlabel('x')
    ax.set_ylabel('')
    ax.set_title(f"Plot of u_t = -u_x , u(0, x) = exp(-x) at t = {t_choice}")
    plt.show()

def test_2D_wave_equation():
    # The goal of this function is to produce error plots in the 2D wave equation case.
    # We will consider the following mD wave equation:
    # 
    # \partial u_1/\partial t + \partial u_1/\partial x + \par = 0  
    # \partial u_2/\partial t + \partial u_1/\partial x = 0

    rng = Random()

    t_cutoff = 0.3

    td = defaultdict(float)

    #u_1(0, x) = 0.5e^{-x1} - 0.5 e^{-2(x1 + x2)}
    #u_2(0, x) = 1.5e^{-x1} - 2.5 e^{-(x1 + x2)}
    #F_1(s) = 1
    #F_2(s) = 1
    td[(1, 1, 1, 0)] = 0.5     # 0.5 exp(- x1)
    td[(1, 1, 0, 1)] = 0.5     # 0.5 exp(- x2)
    td[(1, 2, 1, 0)] = -0.5      # -0.5 exp(-x1)
    td[(1, 2, 0, 2)] = 1.5     # 1.5 exp(- 2 x2)
    td[(2, 1, 0, 0)] = 1
    td[(2, 2, 0, 0)] = 1

    #True u
    def true_u(t, x):
        x1 = x[0]
        x2 = x[1]
        u1 = 0.5 * exp(-x1 + t) + 0.5 * exp(-x2 + t)
        u2 = -0.5 * exp(-x1 + t) + 1.5 * exp(-2 * x2 + 2 * t)
        return [u1, u2]

    x = [1.0, 1.0]
    m = len(x)
    t = t_cutoff/2.0
    means = []
    cis = []
    begin_val = 1
    end_val = 7
    Ntree_range = np.logspace(begin_val, end_val, num = end_val - begin_val + 2, dtype = int)
    for Ntrees in Ntree_range:
        #Use 20 samples to get the average erroraverage
            diff = []
            for i in range(20):
                u_apprx = ApproximateSolution(rng, x, Ntrees, t_cutoff, td)
                L1_diff = sum([abs(u_apprx(t)[i] - true_u(t, x)[i]) for i in range(m)])
                diff.append(L1_diff*L1_diff)
            means.append(np.mean(diff))
            var = np.var(diff, ddof = 1)
            cis.append(1.96 * var/np.sqrt(len(diff)))
    
    sqrt_means = [np.sqrt(m) for m in means]
    sqrt_means = np.array(sqrt_means)


    #plt.style.use('seaborn-darkgrid')
    fig, ax = plt.subplots(figsize=(3.4, 2.2), constrained_layout = True)
    ax.plot(Ntree_range, sqrt_means, marker = 'o', linewidth=2, label=r'$(\mathbb{E}|\mathbf{u} - \tilde{\mathbf{u}}_N|_1^2)^{1/2}$ estimate')
    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.plot(Ntree_range, [0.68/float(np.sqrt(N)) for N in Ntree_range], '--', linewidth=2, label=r'1/$\sqrt{N}$')

    # Customize labels and title
    ax.set_xlabel(r'$N_{\mathrm{trees}}$')
    ax.set_title(r'Convergence for a 2D wave equation system', fontsize=9)

    # Customize legend
    ax.legend(fontsize=8, frameon=True)

    plt.savefig("2D_linear_error_simulation.pdf", bbox_inches = "tight", transparent = True)
    # Adjust layout for clarity
    plt.tight_layout()

    # Display the plot
    plt.show()

def test_high_dimensions(dim):
    # We consider the system of PDEs where F_1(u) = u_1, F_k(u) = 1 for k >= 2.
    # with initial condition u_k(0, x) = (-1)^k 0.5 exp(-(x_1 + ... + x_m))
    rng = Random()

    t_cutoff = 0.1
    Ntrees = 10000
    td = defaultdict(float)

    # This is the initial condition
    for k in range(1, dim + 1):
        td[tuple([1, k] + [1] * dim)] = 0.5 * (-1)**k
    # This is F
    td[tuple([2, 1, 1] + [0] * (dim - 1))] = 1.0
    for k in range(2, dim + 1):
        td[tuple([2, k] + [0] * dim)] = 1.0

    x = [1.0] * dim
    u_appr = ApproximateSolution(rng, x, Ntrees, t_cutoff, td)
    print(f"Solution computed succesfully.")
            
def test_high_dimensions_2(dim):
    # We consider the system of PDEs where F_1(u) = u_1, F_k(u) = 1 for k >= 2.
    # with initial condition u_k(0, x) = (-1)^k 0.5 exp(-x_k)
    rng = Random()

    t_cutoff = 0.001
    Ntrees = 1000
    td = defaultdict(float)

    # This is the initial condition
    for k in range(1, dim + 1):
        alpha = [0]*dim
        alpha[k - 1] = 1
        td[tuple([1, k] + alpha)] = 0.5 * (-1)**k
    # This is F
    td[tuple([2, 1, 1] + [0] * (dim - 1))] = 1.0
    for k in range(2, dim + 1):
        td[tuple([2, k] + [0] * dim)] = 1.0

    #x = [1.0] + [0.0]*(dim - 1)
    x =[1.0]*dim
    u_appr = ApproximateSolution(rng, x, Ntrees, t_cutoff, td)
    print(f"Solution computed succesfully.")

def test_high_dimensions_3(dim):
    # We consider the system of PDEs where F_1(u) = u_1, F_k(u) = 1 for k >= 2.
    # with initial condition u_k(0, x) = (-1)^k 0.5 exp(-x_k)
    rng = Random()

    t = 0.1
    Ntrees = 100000
    td = defaultdict(float)

    # This is the initial condition
    for k in range(1, dim + 1):
        alpha = [0]*dim
        alpha[k - 1] = 1
        td[tuple([1, k] + alpha)] = 0.5 * (-1)**k
    # This is F
    td[tuple([2, 1, 1] + [0] * (dim - 1))] = 1.0
    for k in range(2, dim + 1):
        td[tuple([2, k] + [0] * dim)] = 1.0

    #x = [1.0] + [0.0]*(dim - 1)
    x =[1.0]*dim
    u_appr = AlternateApproximateSolution(rng, x, Ntrees, t, td)

    fig, ax = plt.subplots(1, 2, figsize=(3.8, 2.2), constrained_layout=True)

    # plot 0
    ax[0].plot(range(Ntrees), u_appr[0], label = r'$\tilde{u}_1(t, \mathbf{x})$', linewidth = 1.2)
    ax[0].set_xscale('log')
    ax[0].set_xlabel(r'$N_{\mathrm{trees}}$')
    ax[0].set_title(r'$\tilde{u}_1(t, \mathbf{x})$')

    # plot 1
    ax[1].plot(range(Ntrees), u_appr[1], label = r'$\tilde{u}_2(t, \mathbf{x})$', linewidth = 1.2)
    ax[1].set_xscale('log')
    ax[1].set_xlabel(r'$N_{\mathrm{trees}}$')
    ax[1].set_title(r'$\tilde{u}_2(t, \mathbf{x})$')
    
    fig.suptitle(f"Stabilization plot for a {dim} dimensional system", fontsize = 9)
    plt.savefig("stabilizationHighDimensions.pdf", bbox_inches = "tight", pad_inches = 0.02, dpi = 300, transparent = True)
    plt.show()

    print(f"Solution computed succesfully.")

if __name__ == "__main__":
    test_2D_wave_equation()