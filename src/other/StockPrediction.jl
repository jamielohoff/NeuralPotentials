using Flux, Plots
using CSV, Statistics, DataFrames

### we want to predict the -close- field
stockdata = CSV.read(joinpath(@__DIR__,"nasdaq.csv"), delim=',', DataFrame)

function generatedata(data, split, inputsize)
    len = length(data)
    split_index = trunc(Int, split * len)
    train_labels = data[4:split_index] ./ maximum(data)
    test_labels = data[split_index:end] ./ maximum(data)

    train_features = data[1:split_index-inputsize] ./ maximum(data)
    test_features = data[split_index-inputsize:end-2*inputsize+1] ./ maximum(data)
    for i in 2:inputsize
        _train = data[i:split_index-inputsize+i-1] ./ maximum(data)
        _test = data[split_index-inputsize + i-1:end-2*inputsize+i] ./ maximum(data)
        train_features = hcat(train_features, _train)
        test_features = hcat(test_features, _test)
    end
    return train_features, train_labels, test_features, test_labels
end

train_features, train_labels, test_features, test_labels = generatedata(stockdata.Close, 0.8, 3)

X = reshape(train_features, 3, :)
Y = reshape(train_labels, 1, :)

println(size(X), size(Y))

cb = function ()
    println("test")
end

rnn = Flux.RNN(3, 1)
loss(x, y) = sum(abs2, rnn.(x) .- y)

println(loss(train_features[1,:], train_labels[1]))

ps = Flux.params(rnn)

dataloader = Flux.Data.DataLoader((X, Y), batchsize=1, shuffle=true)
opt = ADAM(1e-2)
epochs = 100

for epoch in 1:epochs
    println("epoch: ", epoch, " of ", epochs)
    for (features, labels) in dataloader
        l = 0.0
        println(features, labels)
        println(loss(features, labels))
        gs = gradient(ps) do 
            l = loss(features, labels)
        end
        Flux.update!(opt, ps, gs)
        cb()
    end
end

