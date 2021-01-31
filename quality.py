from .xharpy import calc_f, construct_values
import numpy as np

def calculate_quality_indicators(construction_instructions, parameters, fjs, cell_mat_m, index_vec_h, intensities, stderr):
    xyz, uij, cijk, dijkl, occupancies = construct_values(parameters, construction_instructions, cell_mat_m)

    structure_factors = calc_f(
            xyz=xyz,
            uij=uij,
            cijk=cijk,
            dijkl=dijkl,
            occupancies=occupancies,
            index_vec_h=index_vec_h,
            cell_mat_f=cell_mat_f,
            symm_mats_vecs=symm_mats_vecs,
            fjs=fjs
    )

    f_obs = np.sign(intensities) * np.sqrt(np.abs(intensities))
    f_obs_safe = np.array(f_obs)
    f_obs_safe[f_obs_safe == 0] = 1e-20
    sigma_f_obs = 0.5 * stderr / np.abs(f_obs_safe)

    r_f = np.sum(np.abs(f_obs - np.sqrt(parameters[0]) * np.abs(structure_factors))) / np.sum(np.abs(f_obs))

    r_f2 = np.sum(np.abs(intensities - parameters[0] * np.abs(structure_factors)**2)) / np.sum(intensities)

    wr_f2 = np.sqrt((intensities - parameters[0] * np.abs(structure_factors)**2) / np.sum(intensities**2 / stderr**2))