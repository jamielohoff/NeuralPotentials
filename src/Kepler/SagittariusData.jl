module SagittariusData
    using DataFrames, CSV, Plots, Zygote

    # conversion factor from km/s to pc/yr
    const kmstopcyr = 1.022e-6
    # conversion factor from microarcseconds to radians
    const mastorad = 4.8481368e-9 
    # distance of Sagittarius A* in parsec
    const D_Astar = 8178 
    # right ascension and declination of Sagittarius A* in radians
    const RA_Astar = 266.4167 * π/180
    const DEC_Astar = -29.0078 * π/180
    const starlist = ["S1", "S2", "S4", "S8", "S9", "S12", "S13", "S14", "S17", "S18", "S19", "S21", 
                        "S24", "S31", "S38", "S54", "S55"]

    """
    Function to convert from microarcseconds to radians.

    Arguments: 
    1. `mas`: Array containing positions in microarcseconds
    """
    function toradian(mas::AbstractArray)
        return mastorad .* mas
    end

    """
    Function to calculate the angle enclosed by the vector (RA, DEC) with the right ascension axis (DEC = 0).

    Arguments:
    1. `RA`: Array containing the right ascensions of an object orbiting a central mass.
    2. `DEC`: Array containing the declinations of an object orbiting a central mass.
    """
    function getangle(RA::AbstractArray, DEC::AbstractArray)
        return mod.(atan.(DEC, RA), 2π) # angle should be in the interval (0, 2π)
    end

    """
    Function to calculate the angular distance of a star at (RA, DEC) from
    the black hole Sagittarius A* in the center of our galaxy at (RA_Astar, DEC_Astar).
    The concept of angular distance is explained @ https://en.wikipedia.org/wiki/Angular_distance.

    Arguments:
    1. `RA`: Array containing the right ascensions of an object orbiting Sagittarius A*.
    2. `DEC`: Array containing the declinations of an object orbiting Sagittarius A*.
    """
    function angulardistance(RA::AbstractArray, DEC::AbstractArray)
        # cosθ = sin(DEC_Astar) * sin.(DEC) .+ cos(DEC_Astar) * cos.(DEC) .* cos.(RA_Astar .- RA)
        cosθ = cos(DEC_Astar) .* cos.(DEC) .+ sin(DEC_Astar) .* sin.(DEC) .* cos.(RA_Astar .- RA)
        return acos.(cosθ)
    end

    """
    Function that loads the data from a star of the nuclear star cluster around Sagittarius A*.
    It outputs a DataFrame that contains the right ascensions and declinations under the keywords RA and DEC,
    as well as the corresponding errors with the keywords RA_err and DEC_err.
    The function assumes that the data is saved as a CSV file where the fields are separated by semicolons(;).
    When the velocities = true flag is set, the function will output two dataframes containing the positional 
    and velocity data of the star. Otherwise, it will just give the position.

    Arguments:
    1. `file`: String that specifies the path to the CSV file.
    2. `designation`: Astronomical designation of the star, e.g. S2, S38 or S55.
    3. `timestamps`: If set to true, includes the timestamp of the measurement in fractional years.
    4. `velocities`: If set to true, includes the velocities in pc/yr.
    """
    function loadstar(file::String, designation::String; timestamps=false, velocities=false)  
        @assert(designation ∈ starlist)
        sagittarius_data = CSV.read(file, delim=';', DataFrame)
        select!(sagittarius_data, Not([:Flag, :Inst]))

        data = []
        if timestamps
            data = sagittarius_data[3:end, filter(x -> x in("Date", "oRA-"*designation, 
                                                                    "e_oRA-"*designation, 
                                                                    "oDE-"*designation, 
                                                                    "e_oDE-"*designation), names(sagittarius_data))]
            dropmissing!(data)
            rename!(data, ["t", "RA", "RA_err", "DEC", "DEC_err"])
            data = parse.(Float32, data)
            data.t = data.t .- 1992.0 # normalize timestamps such that they start from year 0
        else
            data = sagittarius_data[3:end, filter(x -> x in("oRA-"*designation, "e_oRA-"*designation, "oDE-"*designation, "e_oDE-"*designation), names(sagittarius_data))]
            dropmissing!(data)
            rename!(data, ["RA", "RA_err", "DEC", "DEC_err"])
            data = parse.(Float32, data)
        end
        if timestamps && velocities
            velocitydata = sagittarius_data[3:end, filter(x -> x in("Date", "RV-"*designation, "e_RV-"*designation), names(sagittarius_data))]
            dropmissing!(velocitydata)
            rename!(velocitydata, ["t", "RV", "RV_err"])
            velocitydata = parse.(Float32, velocitydata)

            velocitydata.RV = kmstopcyr .* velocitydata.RV
            velocitydata.RV_err = kmstopcyr .* velocitydata.RV_err
            velocitydata.t = velocitydata.t .- 1992.0 # normalize timestamps such that they start from year 0
            return data, velocitydata
        else
            return data
        end
    end

    """
    Function that converts the (RA,DEC) positions of a star at the sky into polar coordinates.
    To compute the angular distance, we need the absolute position of the night sky.
    Therefore, we have to add the position of Sagittarius A* to the given right ascensions 
    and declinations as they are given relative to the black hole.
    Distances in r-direction are given in parsec, while the angle ϕ is given in radians.
    The function assumes that the star belongs to the nuclear star cluster around Sagittarius A*.

    Arguments:
    1. `star`: DataFrame outputted by the function loadstar(...) that contains the trajectory of a specific star.
    """
    function orbit(star::DataFrame)
        ra = toradian(star.RA) .+ RA_Astar
        dec = toradian(star.DEC) .+ DEC_Astar

        ϕ = getangle(toradian(star.RA), toradian(star.DEC))
        # println("θ: ", angulardistance(ra, dec))
        r = D_Astar .* tan.(angulardistance(ra, dec))

        ra_err = toradian(star.RA_err)
        dec_err = toradian(star.DEC_err)

        x_err = zeros(size(r,1)) # sqrt.(D_Astar^2 .+ r.^2) .* tan.(ra_err)
        y_err = zeros(size(r,1)) # sqrt.(D_Astar^2 .+ r.^2) .* tan.(dec_err)

        traj = hcat(r, ϕ, star.t, x_err, y_err)
        return DataFrame(traj, ["r", "ϕ", "t", "x_err", "y_err"])
    end

    """
    Function that returns the coordinates of Sagittarius A*,
    such that we can center the coordinate system around the black hole.
    Outputs a DataFrame containing the the x- and y-offsets, 
    i.e. the position of Sagittarius A*.
    """
    function offsets()
        offsets = DataFrame(x = Real[], y = Real[])
        ϕ_Astar = getangle([RA_Astar], [DEC_Astar])
        R_Astar = sqrt.(RA_Astar.^2 .+ DEC_Astar.^2)# D_Astar * tan.(angulardistance([RA_Astar], [DEC_Astar]))

        x_offset_Astar = R_Astar .* cos.(ϕ_Astar)
        y_offset_Astar = R_Astar .* sin.(ϕ_Astar)
        push!(offsets, [x_offset_Astar[1], y_offset_Astar[1]])
        return offsets
    end 

    """
    Function to center the orbit in polar coordinates such that Sagittarius A* is 
    at polar coordinates (r=0,ϕ=0). Returns a DataFrame containing the centered coordinates 
    in polar and cartesian coordinates.

    Arguments:
    1. `star`: DataFrame containing data about a star from the nuclear star cluster.
    2. `sortby`: Sort by the given field, e.g. :ϕ or :t.
    """
    function centerorbit(star::DataFrame; sortby=:t)
        offset = offsets()
        x = star.R .* cos.(star.ϕ) .- offset.x
        y = star.R .* sin.(star.ϕ) .- offset.y

        r = star.r # sqrt.(x.^2 + y.^2)
        ϕ = getangle(x, y)

        traj = hcat(r, ϕ, x, star.x_err, y, star.y_err, star.t)
        return sort!(DataFrame(traj, ["r", "ϕ", "x", "x_err", "y", "y_err", "t"]), [sortby])
    end

    """
    Function to calculate the reduced χ² statistical measure 
    for a given model prediction, groundtruth and variance/error.
    The dataframe which contains the experimental data has to have two keys, :mu and :me,
    which contain the measurements and their respective standard errors.
    
    Arguments:
    1. `model`: Array that contains the model predictions. The first row must contain the x-postitions 
                and the second row must contain the y-postitions. 
    2. `star`: Dataframe which contains the data of a star in the Sagittarius A* nuclear cluster.
    3. `nparams`: The number of parameters of the model.
    """
    function reducedχ2(model::AbstractArray, star::DataFrame, nparams::Integer)
        n = size(model, 2)
        return (sum(abs2, (model[1,:] .- star.x) ./ star.x_err ) + sum(abs2, (model[2,:] .- star.y) ./ star.y_err)) / (n - nparams)
    end

    """
    Function to load the dataset that contains the orbital parameters 
    as calculated by Gilessen et al. in (https://arxiv.org/abs/1611.09144).

    Arguments:
    1. `file`: Path that points to the CSV file containing the 
    2. `designation`: Name of the star, for example S2.
    """
    function loadorbitalelements(file::String, designation::String)
        @assert(designation ∈ starlist)
        sagittarius_data = CSV.read(file, delim=';', DataFrame)
        select!(sagittarius_data, Not([:f_Star, :SpT, :Kmag, :Orb, :SimbadName]))
        data = sagittarius_data[4:end, :]
        dropmissing!(data)
        rename!(data, ["Star", "a", "a_err", "e", "e_err", "i", "i_err", "Ω", "Ω_err", "ω", "ω_err", "Tp", "Tp_err", "Per", "Per_err", "r", "_RA", "_DE"])
        data = data[findall(in([designation]), data.Star), :]
        select!(data, Not([:Star]))
        data = parse.(Float32, data)
        return data
    end

    """
    Function that rotates an array of 3d points around the axis n⃗ by the angle α.
    It outputs three lists containing the x, y and z coordinates of all the points.

    Arguments:
    1. `x`: List of x coordinates.
    2. `y`: List of y coordinates.
    3. `z`: List of z coordinates.
    4. `n`: Axis of rotation.
    5. `α`: Rotation angle.
    """
    function rotation(x::AbstractArray, y::AbstractArray, z::AbstractArray, n::AbstractArray, α::Real)
        @assert size(n,1) == 3
        @assert size(x) == size(y) == size(z)
        X = ( n[1]^2*(1 - cos(α)) + cos(α) ) .* x + ( n[1]*n[2]*(1 - cos(α)) - n[3]*sin(α) ) .* y + ( n[1]*n[3]*(1 - cos(α)) + n[2]*sin(α) ) .* z
        Y = ( n[1]*n[2]*(1 - cos(α)) + n[3]*sin(α) ) .* x + ( n[2]^2*(1 - cos(α)) + cos(α) ) .* y + ( n[2]*n[3]*(1 - cos(α)) - n[1]*sin(α) ) .* z
        Z = ( n[1]*n[3]*(1 - cos(α)) - n[2]*sin(α) ) .* x + ( n[2]*n[3]*(1 - cos(α)) + n[1]*sin(α) ) .* y + ( n[3]^2*(1 - cos(α)) + cos(α) ) .* z
        return X, Y, Z
    end

    """
    Function that rotates an array of 3d points around the axis n⃗ by the angle α.
    It outputs three lists containing the x, y and z coordinates of all the points.

    Arguments:
    1. `x`: Vector of x coordinates.
    4. `n`: Axis of rotation.
    5. `α`: Rotation angle.
    """
    function rotation(x::AbstractArray, n::AbstractArray, α::Real)
        @assert size(n,1) == 3
        @assert size(x,1) == 3
        X = ( n[1]^2*(1 - cos(α)) + cos(α) ) .* x[1] + ( n[1]*n[2]*(1 - cos(α)) - n[3]*sin(α) ) .* x[2] + ( n[1]*n[3]*(1 - cos(α)) + n[2]*sin(α) ) .* x[3]
        Y = ( n[1]*n[2]*(1 - cos(α)) + n[3]*sin(α) ) .* x[1] + ( n[2]^2*(1 - cos(α)) + cos(α) ) .* x[2] + ( n[2]*n[3]*(1 - cos(α)) - n[1]*sin(α) ) .* x[3]
        Z = ( n[1]*n[3]*(1 - cos(α)) - n[2]*sin(α) ) .* x[1] + ( n[2]*n[3]*(1 - cos(α)) + n[1]*sin(α) ) .* x[2] + ( n[3]^2*(1 - cos(α)) + cos(α) ) .* x[3]
        return Array([X, Y, Z])
    end


    """
    Function that corrects the phase when we have orbits that go for more than one revolution.
    One full evolution takes a polar angle of 2π. The for each following revolution, we add another 2π to
    the polar angle.

    Arguments:
    1. `ϕ`: Array that contains the polar angle of the trajectory/orbit. 
    """
    function correctphase(ϕ::AbstractArray)
        buf = Zygote.Buffer(zeros((size(ϕ,1),1)))
        phase = 0.0
        for j ∈ 1:size(ϕ,1)-1
            buf[j,1] = ϕ[j] + phase
            if abs(ϕ[j+1] - ϕ[j]) > 5.0
                phase += 2π
            end
        end
        buf[end,1] = ϕ[end] + phase
        res = copy(buf)
        return res[:,1]
    end
    
    """
    Function that rotates a trajectory in the x-y-plane around an axis in the x-y-plane...

    Source and explanation of the angles: 
    https://www.narom.no/undervisningsressurser/sarepta/rocket-theory/satellite-orbits/introduction-of-the-six-basic-parameters-describing-satellite-orbits/


    Arguments:
    1. `angles`: Array that contains the Keplerian orbital elements, i.e. the three angles inclination ι, 
                 longitude of the ascending node Ω and argument of periapsis ω.
    2. `r`: Radial coordinate of the trajectory in the x-y-plane.
    3. `ϕ`: Polar angle of the original trajectory in the x-y-plane.
    """
    function transform(angles::AbstractArray, r::AbstractArray, ϕ::AbstractArray)
        ι = mod(angles[1],2π)
        Ω = mod(angles[2],2π)
        ω = mod(angles[3],2π)
    
        x = r.*cos.(ϕ)
        y = r.*sin.(ϕ)
        z = zeros(size(x))
    
        n = [cos(Ω), sin(Ω), 0.0]
        X, Y, Z = rotation(x, y, z, n, ι)
    
        m = [sin(ι)*sin(Ω), -sin(ι)*cos(Ω), cos(ι)]
        _X, _Y, _Z = rotation(X, Y, Z, m, mod(ω + Ω,2π))
    
        φ = mod.(atan.(_Y,_X), 2π)
        φ = SagittariusData.correctphase(φ)
        R = sqrt.(_X.^2 + _Y.^2)
        return R, φ
    end
    
    
    """
    Function that...

    Source and explanation of the angles: 
    https://www.narom.no/undervisningsressurser/sarepta/rocket-theory/satellite-orbits/introduction-of-the-six-basic-parameters-describing-satellite-orbits/

    Arguments:
    1. `angles`: Array that contains the Keplerian orbital elements, i.e. the three angles inclination ι, 
                 longitude of the ascending node Ω and argument of periapsis ω.
    2. `r`: Radial coordinate of the trajectory in the x-y-plane.
    3. `ϕ`: Polar angle of the original trajectory in the x-y-plane.
    """
    function inversetransform(angles::AbstractArray, r::AbstractArray, ϕ::AbstractArray)
        ι = mod(angles[1],2π)
        Ω = mod(angles[2],2π)
        ω = mod(angles[3],2π)
        
        x = r.*cos.(ϕ)
        y = r.*sin.(ϕ)
        z = r.*sin.(ϕ .- Ω).*tan(ι)
    
        m = -[sin(ι)*sin(Ω), -sin(ι)*cos(Ω), cos(ι)]
        X, Y, Z = rotation(x, y, z, m, mod(ω + Ω,2π))
    
        n = -[cos(Ω), sin(Ω), 0.0]
        _X, _Y, _Z = rotation(X, Y, Z, n, ι)

        φ = mod.(atan.(_Y,_X), 2π)
        φ = SagittariusData.correctphase(φ)
        R = sqrt.(_X.^2 + _Y.^2)
        return R, φ
    end
end

