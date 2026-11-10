# papers/ — reading for part 01 of the assignment (discussed in the interview, nothing submitted)

| File | Paper |
|------|-------|
| `3D_Gaussian_splatting.pdf` | Kerbl et al., *3D Gaussian Splatting for Real-Time Radiance Field Rendering*, SIGGRAPH 2023 |
| `HYbrid_particle-voxel.pdf` | Wang et al., *Hybrid Particle-Voxel Dynamics for Multimodal 3D World Simulation*, arXiv 2026 |

## How they relate to this project

- **Gaussian Splatting** would replace the box proxy with a photorealistic 3D reconstruction of the ball. The assignment does not assess reconstruction, so we use a box.
- **Particle-voxel dynamics** is the full-scale version of what we do: physical parameters of deformable objects identified from visual observation. We reduce it to a single degree of freedom (spring-mass-damper) with two parameters, `k` and `c`.
