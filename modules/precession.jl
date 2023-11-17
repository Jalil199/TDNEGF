module precession
#export PrecSpin, update!
### Libraries
# include("parameters.jl")
# using .parameters
include("get_parameters.jl")
using .get_parameters

mutable struct PrecSpin
    """ This mutable structure act like a class 
    in python, it defines an object with the 
    characteristics of a precessing spin
    """
    i::Int64 #   
    axis_phi::Float64 
    axis_theta::Float64 
    phi_zero::Float64 
    theta_zero::Float64 
    start_time::Float64 
    T::Float64 
    s::Vector{Float64} 
    ### Initial values of PrecSpin .
    function PrecSpin(i=0,axis_phi=0.0,axis_theta=0.0,
            phi_zero=0.0,theta_zero=0.0,start_time=0.0
            ,T=1.,s=[0.,0.,1.])
        new(i,axis_phi,axis_theta,phi_zero,theta_zero,start_time,T,s)
    end
end

function update!(this, time)#(this::PrecSpin, time  )
    """ This function  update the magnetic moment associated
    to the mutable structure PrecSpin 
    
    parameters:
    ----------
    this: mutable structure 
    contain an structure with the characteristics of a spin 
    time: Float64
    time where the spin is evaluated 
    
    returns:
    -------
    Update the strucure associated to a  precessing spin 
    """
    if time >= this.start_time
        t = time - this.start_time
    else
        t = 0.0
    end
    omega = 2*pi / this.T
    otheta = pi* this.theta_zero/180.
    ophi = pi*this.phi_zero /180. ##########
    # compute spin position for precession along the z-axis
    sz = cos(otheta)
    sx = cos(ophi+ omega*t)*sin(otheta)
    sy = sin(ophi+ omega*t)*sin(otheta)
    #Now rotate along y 
    atheta = pi * this.axis_theta/ 180.
    aphi = pi * this.axis_phi / 180. 
    sx = sx*cos(atheta) - sz* sin(atheta)
    sz = sx* sin(atheta) + sz*cos(atheta)
    #No rotate along the z 
    sx = sx*cos(aphi) + sy*sin(aphi)
    sy = -sx*sin(aphi) + sy*cos(aphi)
    # sx = 0.
    # sy = 0. 
    # sz = 1. + sin(omega*t)*0.5 
    this.s .= [sx, sy, sz]
    nothing
end


end