using LinearAlgebra, Zygote, Printf
using StatsBase: mean
using Parameters
using Distributions, ProgressBars
using Random
using BSON

### Below you will implement your linear regression object from scratch.
### you will train it using gradient descent with momentum, 
### you will also imply some penalty methods as well.

#### ----- ###
#### Before getting started you should write your student_number in integer format
const student_number::Int64 = 61819478756  ## <---replace 0 by your student_number 
### ---- ###

### This hw is a bit harder than the usual ones, so you may want to
### take a look at --Julia official website 
### Before you get started, --- you gotta look at do end syntax of Julia
### Instead of mean square loss we shall use Huber loss which
### sometimes performs better when your dataset contains some out of distribution points
### We split it into two parts the first one is for scalars the second one is for vectors
### remember that you will multiple dispatch!!!


function huber_loss(y_pred::T, y_true::T; δ::Float64 = 0.1) where T <: Real
    huber_loss(a,δ)= abs(a) <= δ ? 1/2*a^2 : δ*(abs(a)-1/2*δ)
    return  huber_loss(y_pred - y_true, δ)
end

function huber_loss(y_pred::AbstractArray{T}, y_true::AbstractArray{T}) where T <: Real
    return mean(huber_loss.(y_pred, y_true))
end

function unit_test_huber_loss()::Bool
    Random.seed!(0)
    @assert huber_loss(1.0,1.0) == 0.0
    @assert huber_loss(1.0,2.0; δ = 0.9) == 0.49500000000000005
    @assert isapprox(huber_loss(randn(100,100),randn(100,100)), 0.10791842, atol = 1e-2) 
    @assert isapprox(huber_loss(randn(100),randn(100)), 0.107945, atol = 1e-2)
    @info "You can not stop now comrade!!! jump to the next exercise!!!"
    return 1
end

## See you have implemented huber_loss() well??
unit_test_huber_loss()

### create a roof for the logistic regression LogisticClassifier
abstract type LinearRegression end 
mutable struct linear_regression <: LinearRegression
    ## This part is given to you!!!
    ## Realize that we have fields: θ and bias.
    θ::AbstractVector
    bias::Real
    linear_regression(n::Int64) = new(0.004*randn(n), zero(1.0))
end


### write the forwardpass function 
function (lr::linear_regression)(X::Matrix{T}) where T<:Real
    ## This dude is the forward pass function!!!
    return X * lr.θ .+ lr.bias
end


function unit_test_forward_pass()::Bool
    try
        linear_regression(20)(randn(10,20))  
    catch ERROR
        error("SEG-FAULT!!!!")
    end
    @info "Real test started!!!!"
    for i in ProgressBar(1:10000)
        sleep(0.001)
        lr = linear_regression(3)
        x = randn(2,3)    
        @assert lr(x) == x*lr.θ .+ lr.bias 
    end
    @info "Oki doki!!!"
    
    return 1
end

### Let's give a try!!!!!!
unit_test_forward_pass()
## we shall now implement fit! method!!!
## before we get ready run the next 5 lines to see in this setting grad function returns a dictionary --named tuple actually!!!:
#=
lr = linear_regression(10)
grad = Zygote.gradient(lr) do lr
    norm(lr.θ) + lr.bias
end
=#

@inline function fit!(lr::linear_regression, 
    X::AbstractMatrix, 
    y::AbstractVector; 
    learning_rate::Float64 = 0.00001, 
    max_iter::Integer = 5,
    λ::Float64 = 0.01, ## Penalty term magnitude!!!
    β::Float64 = 0.9) ## β here is the β in gradient descent with momentum!!!
    ##
    
    ## You should create a dictionary for the velocity term!!!
    ## before getting into the for loop 
    ## Please see the slides and the related .jl files 


    velocity = Dict{Int, Array{Float64}}()

    function update_grads!(lr::LinearRegression, learning_rate::Float64, grads)
        
        lr.θ .-= learning_rate * grads[:θ] 
        lr.bias -= learning_rate * grads[:bias]  
    end


    for i in 1:max_iter
        ## grab the gradients
    
       
        ## Here you will update_the weights, 
        ## There will be one more for loop running  over keys of the grad -- here!!!
        ## Do not forget to update the velocity term!!!
        ## Search for setfield!, setindex! and getfield functions.
        lr = linear_regression(10)
        val, grad = Zygote.withgradient(lr) do lr
            norm(lr.θ)^2 + lr.bias^2
        end


        
        if i % 100 == 0
            println("The loss is $(val)")
        end
        
    end
end


## Let's give a try!!!
lr = linear_regression(20)
X = randn(100, 20)
y = randn(100)
fit!(lr, X, y; learning_rate = 0.00001, max_iter = 10000)
### Things seem to work fine if the loss decreases --Give some attention here!!!

function unit_test_for_fit()
    Random.seed!(0)
    lr = linear_regression(20)
    X = randn(100, 20)
    y = randn(100)
    fit!(lr, X, y; learning_rate = 0.0001, max_iter = 10000, λ = 0.1)
    @assert norm(lr.θ)^2 + lr.bias^2 < 0.01 "Your penalty method does not work!!!"
    @assert mean((lr(X) - y).^2) < 1.2 "Yo do not fit perfectly!!!!"
    @info "Okito dokito buddy!!!"
    return 1
end


## Run next line to see what happens??? ##
unit_test_for_fit()










## No need to run below!!!
if abspath(PROGRAM_FILE) == @__FILE__
    G::Int64 = unit_test_huber_loss()  + unit_test_forward_pass() + unit_test_for_fit()
    dict_ = Dict("std_ID"=>student_number, "G"=>G)
    try
        BSON.@save "$(student_number).res" dict_ 
        catch Exception 
            println("something went wrong with", Exception)
    end
end
