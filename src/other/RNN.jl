using DifferentialEquations, Flux, DiffEqFlux, Zygote
using Plots, LinearAlgebra, Statistics, LaTeXStrings, ColorSchemes,Measures
include("../MechanicsDatasets.jl")
using .MechanicsDatasets

theme(:mute)
Plots.resetfontsizes()
Plots.scalefontsizes(2)

### Creation of synthethetic data---------------------- #
tspan = (0.0, 8.5)
t = Array(range(tspan[1], tspan[2], length=256))
true_u0 = [3.0, 0.0] # contains x0 and dx0
true_p = [2.0]
stderr = 0.1
# Define potential and get dataset
V0(x,p) = 0.5*p[1]*x^2 
data = MechanicsDatasets.potentialproblem1D(V0, true_u0, true_p, t, addnoise=true, σ=stderr)
### End ------------------------------------------------ #

function train(num_in::Int, num_hidden::Int, epochs::Int, batchsize::Int)
    x = rand(num_in)
    h = rand(num_hidden)

    NN = Chain(
        # Flux.RNNCell(num_hidden, num_in, tanh),
        Dense(num_in + num_hidden, 16, tanh),  # Flux.LSTMCell(num_in + num_hidden, 16), # 
        Dense(16, 16, tanh), 
        Dense(16, 16, tanh),
        Dense(16, num_hidden + num_in)
    )
    ps = Flux.params(NN)

    function rnn(h, x)
        inputs = vcat(h, x)
        result = NN(inputs)
        return result[1:num_hidden], result[num_hidden+1:end]
    end

    model = Flux.Recur(rnn, h)

    function createbatch(data, interval, batchsize)
        @assert size(interval,1) == 2
        @assert interval[1] ≥ data[1,1]
        @assert interval[2] ≤ data[1,end]

        a = findall(x -> x ≥ interval[1], data[1,:])[1]
        b = findall(x -> x ≥ interval[2], data[1,:])[1]
        timelist = []
        datalist = []

        for i ∈ 1:batchsize
            idx = rand(a:b-num_in)
            push!(timelist, data[1,idx:idx+num_in-1])
            push!(datalist, data[2,idx:idx+num_in-1])
        end
        return (timelist, datalist)
    end

    function loss(input, output)
        err = [sum(abs2, model(input[i]) - output[i]) for i ∈ 1:size(input,1)]
        return mean(abs2, err)
    end

    opt = ADAM(1e-3)

    for epoch ∈ 1:epochs
        batch = createbatch(data,(0.0, 6.0),batchsize)
        l = 0.0
        gradient = Flux.gradient(ps) do
            l = loss(batch...)
            # println("Loss at thread ", Threads.threadid(), " : ", l)
            return l
        end
        if l < 0.001
            break
        end
        Flux.update!(opt, ps, gradient)
    end

    function test(model, data)
        prediction = [x for x ∈ model(data[1,1:num_in])]
        for i ∈ 2:size(data[1,:],1) - num_in
            push!(prediction, model(data[1,i:i+num_in-1])[end])
        end
        return prediction
    end
    return test(model, data)
end

repetitions = 512
predictions = []

lk = ReentrantLock()
@time Threads.@threads for rep in 1:repetitions
    println("Starting repetition ", rep, " of ", repetitions, " @ thread ", Threads.threadid())
    pred = train(5, 10, 10000, 8)
    lock(lk)
    push!(predictions, pred)
    unlock(lk)
end

function quantiles(data; bounds=[0.025, 0.975])
    itm = data[1]
    for i in 2:size(data,1)
        itm = hcat(itm, data[i])
    end
    qtl = Float64[0,0]
    for j in 1:size(itm, 1)
        qtl = hcat(qtl, quantile(itm[j,:], bounds))
    end
    return qtl[:,2:end]
end

μ = mean(predictions)
qtls = quantiles(predictions)

lower_CI = qtls[1,:] - μ
upper_CI = qtls[2,:] - μ
CI = [lower_CI, upper_CI]

plt = scatter(data[1,:], data[2,:], 
            legend=:bottomright, 
            label="Harmonic oscillator data",
            xlabel=L"\textrm{Time } t",
            ylabel=L"\textrm{Position } x",
            ylims=(-5.0, 5.0),
            size=(1200, 800),
            margin=5mm,
            markersize=6,
            markerstyle=:cross,
            markerstrokewidth=0,
            foreground_color_minor_grid = "white",
            framestyle=:box,
            color=colorant"#328" # indigo
)
plt = plot!(plt, data[1,1:end-1], μ, ribbon=CI, 
        label="RNN prediction",
        linewidth=3,
        color=colorant"#c67" # rose
)

savefig(plt, "RNN.pdf")

