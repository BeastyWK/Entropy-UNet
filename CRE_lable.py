from ovito.io import import_file
import numpy as np
import warnings
import math
from collections import Counter
warnings.filterwarnings('ignore', message='.*OVITO.*PyPI')

pipeline = import_file(r"rna.dumplammps")


cutoff = 5.5
sigma = 0.125

# Number of bins used for integration:
nbins = int(cutoff / sigma) + 1

# Table of r values at which the integrand will be computed:
r = np.linspace(0.0, cutoff, num=nbins)
rsq = r ** 2



def local_entropy(data,cutoff = 5.5, sigma = 0.125, use_local_density = False):
    # Overall particle density:
    global_rho = data.particles.count / data.cell.volume

    # Initialize neighbor finder:
    finder = CutoffNeighborFinder(cutoff, data)

    # Create output array for local entropy values
    local_entropy = np.empty(data.particles.count)

    # Number of bins used for integration:
    nbins = int(cutoff / sigma) + 1

    # Table of r values at which the integrand will be computed:
    r = np.linspace(0.0, cutoff, num=nbins)
    rsq = r ** 2

    # Precompute normalization factor of g_m(r) function:
    prefactor = rsq * (4 * np.pi * global_rho * np.sqrt(2 * np.pi * sigma ** 2))
    prefactor[0] = prefactor[1]  # Avoid division by zero at r=0.


    # Iterate over input particles:
    for particle_index in range(data.particles.count):

        # Get distances r_ij of neighbors within the cutoff range.
        r_ij = finder.neighbor_distances(particle_index)

        # Compute differences (r - r_ji) for all {r} and all {r_ij} as a matrix.
        r_diff = np.expand_dims(r, 0) - np.expand_dims(r_ij, 1)

        # Compute g_m(r):
        g_m = np.sum(np.exp(-r_diff ** 2 / (2.0 * sigma ** 2)), axis=0) / prefactor
        # g_m += 1e-10

        # Estimate local atomic density by counting the number of neighbors within the
        # spherical cutoff region:
        if use_local_density:
            local_volume = 4 / 3 * np.pi * cutoff ** 3
            rho = len(r_ij) / local_volume
            if rho == 0:
                rho = global_rho
            g_m *= global_rho / rho
        else:
            rho = global_rho


        # Compute integrand:
        integrand = np.where(g_m >= 1e-10, g_m* rsq, rsq)

        local_entropy[particle_index] = 4*np.pi * rho * np.trapezoid(integrand, r)


    return local_entropy




num_frames = pipeline.source.num_frames
print("frames:",num_frames)

shannon_entropy = []
for frame_index in range(0, num_frames):
    # 获取当前帧的数据
    data = pipeline.compute(frame_index)

    densities_rdf = local_entropy(data, cutoff = 2.1, sigma = 0.125, use_local_density = True)[0:84:3]#[2:84:3][:84]



    total_density = np.sum(densities_rdf)



    normalized_densities = [d / total_density for d in densities_rdf]

    #  save lables
    normalized_npy = np.array(normalized_densities).reshape(84,1)
    np.save("lables\\PMF\\" + str(frame_index+30001) + ".npy",normalized_npy)



    shannon_entropy.append(-sum([d * math.log2(d) for d in normalized_densities if d > 0]))





with open(r'Shannon_entropy_RNA.txt', 'w') as file:
    for item in shannon_entropy:
        file.write(f"{item}\n") 
print("OK!")





