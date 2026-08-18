from __future__ import annotations
import json, os
from pathlib import Path
import numpy as np
from mean_field.systems.inas_gasb import Kane4Bundle, MatrixEIConfig, ProjectedFockOperator, solve_reference_subtracted_matrix_ei

nk=3
weights=np.array([0.03,0.07,0.11])
h0=np.empty((4,4,nk),complex)
for ik in range(nk):
    h0[:,:,ik]=np.array([
        [-0.35+0.04*ik, 0.01j, 0.025, -0.008j],
        [-0.01j,-0.30+0.03*ik,0.006j,0.021],
        [0.025,-0.006j,0.22+0.02*ik,-0.012j],
        [0.008j,0.021,0.012j,0.27+0.01*ik],
    ])
phi=np.zeros((nk,2,4,4),complex)
phi[:,0,:,:]=np.eye(4)[None,:,:]/np.sqrt(2)
phi[:,1,:,:]=np.eye(4)[None,:,:]/np.sqrt(2)
bundle=Kane4Bundle(
    k_cart_nm_inv=np.column_stack([np.arange(nk)*0.01,np.zeros(nk)]),
    weights_nm2=weights,
    z_nm=np.array([-0.5,0.5]),
    z_weights_nm=np.ones(2),
    h0_mev=h0,
    micro_wavefunctions=phi,
    provenance={"source":"old-new-generic-api-parity"},
)
green=np.empty((nk,nk,2,2),float)
for i in range(nk):
    for j in range(nk):
        green[i,j]=0.9/(1+abs(i-j))*np.array([[1.0,0.55],[0.55,1.0]])
operator=ProjectedFockOperator.from_bundle(bundle,green,self_cell_description="old-new parity")
result=solve_reference_subtracted_matrix_ei(
    bundle,operator,
    config=MatrixEIConfig(
        temperature_K=0.25,mixing=0.3,precision=1e-11,max_iter=500,
        constraint_policy="kane_poisson_fixed_mu",fixed_mu_mev=0.0,
        normal_ordering_reference_policy="electron_hole_vacuum",
    ),
)
out=Path(os.environ["PARITY_OUTPUT"])
np.savez_compressed(
    out,
    reference_density=result.reference_density,
    noninteracting_density=result.noninteracting_density,
    density_delta=result.density_delta,
    total_density=result.total_density,
    sigma_fock=result.sigma_fock_mev,
    hamiltonian=result.hamiltonian_mev,
    energies=result.energies_mev,
    iter_err=result.run.iter_err,
    iter_energy=result.run.iter_energy,
    iter_oda=result.run.iter_oda,
    scalars=np.array([
        result.reference_mu_mev,result.mu_mev,
        result.one_body_internal_energy_density_mev_nm2,
        result.fock_internal_energy_density_mev_nm2,
        result.total_internal_energy_density_mev_nm2,
        result.entropy_difference_mev_per_K_nm2,
        result.canonical_free_energy_difference_mev_nm2,
        result.grand_potential_difference_mev_nm2,
        result.run.state.diagnostics["final_raw_norm"],
    ]),
)
print(json.dumps({"converged":result.run.converged,"iterations":result.run.iterations,"exit_reason":result.run.exit_reason,"classification":result.classification}))
