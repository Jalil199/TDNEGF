# TDNEGF-Hybrid
TDNEGF-Hybrid is a Julia package that simulates time-dependent quantum transport in open quantum systems with an optional coupling to classical magnetic moments. 
The code implements the TDNEGF method discussed in Ref. [https://doi.org/10.1103/PhysRevApplied.10.054038]. This is a reformulation of the theory of
Ref. [https://doi.org/10.1088/1367-2630/18/9/093044] in terms of wave-vectors, which insures a reduced runtime. In effect, the code solves a system of ordinary differential 
equations (ODEs), made up of the reduced system density matrix (ρS) of the quantum system and other auxiliary quantities, namely the wave-vectors ψ and the 
scalars Ω.  In conclusion, this method aims at numerically solving a (very large ) set of ODEs.
