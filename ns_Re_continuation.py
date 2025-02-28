from firedrake import *
from firedrake.petsc import PETSc
import time

# gmsh -2 -format msh2 -clscale 0.1 mesh.geo -algo front2d -o mesh1.msh -smooth 3
# gmsh -2 -format msh2 -clscale 0.01 mesh.geo -algo front2d -o mesh1.msh -smooth 3
# Load the mesh
mesh = Mesh("./mesh1.msh")

# Define function spaces
V = VectorFunctionSpace(mesh, "CG", 2)  # P2: Quadratic velocity (Continuous Galerkin)
W = FunctionSpace(mesh, "DG", 0)        # P0: Discontinuous constant pressure
Z = V * W  # Mixed function space for velocity and pressure

# Define trial and test functions
up = Function(Z)  # Solution function (velocity and pressure)
u, p = split(up)  # Split into velocity and pressure
v, q = TestFunctions(Z)  # Test functions for velocity and pressure

# Problem parameters
Re_values = [10.0, 100.0, 1000.0, 10000.0]  # Reynolds numbers to solve for

Re = Constant(10.0)  # Define Re as a Constant for dynamic updates


# Define boundary conditions
class BoundaryConditions:
    @staticmethod
    def poiseuille_flow(mesh):
        """
        Parabolic velocity profile for Poiseuille flow in a channel.
        Adjust the profile to match the specific domain geometry.
        """
        y = SpatialCoordinate(mesh)[1]  # Access the y-coordinate
        ymax = 2.0  # Maximum y-coordinate (domain height)
        return as_vector(
            [4.0 * (2.0 - y) * (y - 1) * (y > 1), 0.0]
        )  # Parabolic profile


# Weak form of the Navier-Stokes equations
beta = Constant(0.0)  # Penalty parameter for the Augmented Lagrangian term
# Define the local element size h
h = CellSize(mesh)

# Define SUPG stabilization parameter (tau)
tau = 1.0 / sqrt(
    (4.0 / (h**2)) * (1.0 / Re) + dot(u, u) / h
)

# Residuals
R_momentum = 1.0 / Re * div(grad(u)) + dot(grad(u), u) + grad(p)  # Momentum residual
R_continuity = div(u)  # Continuity residual

# SUPG stabilization term
dx_custom = dx(degree=3)  # Use a reasonable quadrature degree
SUPG = (
    tau * inner(dot(grad(v), u), R_momentum) * dx_custom  # SUPG for momentum
    + tau * inner(div(v), R_continuity) * dx_custom  # SUPG for continuity
)
F = (
    1.0 / Re * inner(grad(u), grad(v)) * dx_custom  # Viscous term
    + inner(dot(grad(u), u), v) * dx_custom  # Convective term
    - inner(p, div(v)) * dx_custom  # Pressure term
    + div(u) * q * dx_custom  # Continuity equation
    # + beta * inner(cell_avg(div(u)), div(v)) * dx_custom      # Augmented Lagrangian penalty term
    + SUPG
)

# Apply boundary conditions
bcs = [
    DirichletBC(
        Z.sub(0), BoundaryConditions.poiseuille_flow(Z.mesh()), 1
    ),  # Inlet with Poiseuille flow
    DirichletBC(Z.sub(0), Constant((0.0, 0.0)), 2),  # No-slip walls
]

# Nullspace for pressure stabilization
nullspace = MixedVectorSpaceBasis(Z, [Z.sub(0), VectorSpaceBasis(constant=True)])
appctx = {"Re": Re, "velocity_space": 0}

# parameters_orig = {
#     "mat_type": "matfree",
#     "snes_monitor": None,
#     "snes_max_it": 100,
#     "ksp_type": "fgmres",
#     # "snes_ksp_ew": None,
#     "snes_dtol": 1e5,                                  # Divergence tolerance
#     "divtol": -1,                                      # Custom divergence tolerance
#     "snes_divtol": -1,                                 # Custom divergence tolerance for SNES
#     "snes_linesearch_type": "bt",                      # Use backtracking line search
#     "ksp_gmres_modifiedgramschmidt": None,
#     "ksp_monitor_true_residual": None,
#     "pc_type": "fieldsplit",
#     "pc_fieldsplit_type": "schur",
#     "pc_fieldsplit_schur_fact_type": "full",
#     "fieldsplit_0_ksp_type": "preonly",
#     "fieldsplit_0_pc_type": "python",
#     "fieldsplit_0_pc_python_type": "firedrake.AssembledPC",
#     "fieldsplit_0_assembled_pc_type": "lu",
#     # "fieldsplit_0_ksp_monitor": None,
#     # "fieldsplit_1_ksp_monitor": None,
#     "fieldsplit_1_ksp_type": "gmres",
#     "fieldsplit_1_ksp_rtol": 1e-4,
#     "fieldsplit_1_pc_type": "python",
#     "fieldsplit_1_pc_python_type": "firedrake.PCDPC",
#     "fieldsplit_1_pcd_Mp_ksp_type": "preonly",
#     "fieldsplit_1_pcd_Mp_pc_type": "lu",
#     "fieldsplit_1_pcd_Kp_ksp_type": "preonly",
#     "fieldsplit_1_pcd_Kp_pc_type": "lu",
#     "fieldsplit_1_pcd_Fp_mat_type": "matfree",
# }
parameters_orig = {
    "snes_type": "anderson",
    "snes_converged_reason": None,
    "snes_monitor": None,
    "snes_max_it": 100000,
    "snes_linesearch_type": "bt",
    "snes_stol": 0.0,
    "npc": {
        "snes_type": "anderson",
        "snes_atol": 0.0,
        "snes_rtol": 0.0,
        "snes_stol": 0.0,
        "snes_max_it": 1000,
        "snes_converged_reason": None,
        # "ksp_monitor": None,
    },
}
# Solve for increasing Reynolds numbers
for i, Re_value in enumerate(Re_values):
    PETSc.Sys.Print(f"Solving for Re = {Re_value}")

    Re.assign(Constant(Re_value))

    # Use the current solution as the initial guess for the next solve
    up.assign(up)  # Explicitly set the initial guess to the current solution

    # Solve the Navier-Stokes equations
    start_time = time.time()
    solve(
        F == 0,
        up,
        bcs=bcs,
        nullspace=nullspace,
        solver_parameters=parameters_orig,
        appctx=appctx,
    )
    end_time = time.time()

    # Log timing
    PETSc.Sys.Print(f"Re = {Re_value} solved in {end_time - start_time:.2f} seconds")

    # Save the solution to file
    u, p = up.subfunctions
    u.rename("Velocity")
    p.rename("Pressure")
    VTKFile(f"solution_Re_{Re_value}.pvd").write(u, p)

    # Use the solution as the initial guess for the next Reynolds number
    if i < len(Re_values) - 1:
        PETSc.Sys.Print(
            f"Using solution for Re = {Re_value} as initial guess for Re = {Re_values[i+1]}"
        )

    # Update Re parameter


PETSc.Sys.Print("All Reynolds number solutions computed successfully.")
