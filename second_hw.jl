### For this HW, you shall implement a very simple version of harmonic regression 
### for datasets that have periodic components. This is what is called approximations
### by trigonometric polynomials (sometimes therefore trigonometric regression)
### --- ###
### Run the following part to see if there is any package that you need to install
### if something goes wrong, watch the error message and install the needed package by 
### Pkg.add("missing_package")
using Pkg
using DataFrames, CSV
using Plots: plot, plot!
using Statistics
using BSON
using Random
using Distributions: DiscreteUniform
using ProgressMeter

#### ----- ###
cd(@__DIR__)
#### Before getting started you should write your student_number in integer format
const student_number::Int64 = 0  ## <---replace 0 by your student_number 
### ---- ###
 
### Let's do an experiment
x = 0:0.1:20
f = x-> (1+2*cos(2*pi*x/3)-3*sin(2*pi*x/5)) + randn()
r = map(f, x);
plot(x, r)
## You see we have a lot of seasonality (periodic components)
## end of experiment

## For such datasets we may have the following model
x_ = 0:0.1:40
x = 0:0.1:20
f_test(x::Real) = (1+2*cos(2*pi*x/3)-3*sin(2*pi*x/5))
r_ = map(f_test, x_);
plot(x, r, label = "Real data")
plot!(x_, r_, label = "Proposal models approximation")
## As you see the model closely approximates the data
## Please notice that we have here a period 15!!!!!
## Therefore it is not coincidence that we have 3 and 5 in the denominators.
## end of proposal model ## 

### Now we shall learn how to find out the coefficients
### 1 2 -3, given any dataset (caveat: the model may not be the best fit for all datasets)
### Now create a mutable struct that we can use for our harmonic polynomial 
mutable struct harmonic_pol
    period::Int
    coeff::Vector{Float64}
end
## for instance run the following two lines:
Random.seed!(0)
pol = harmonic_pol(5, randn(3))
## here the trigonometric polynomial corresponds to:
## p(x) = 0.94 + 0.13*cos(2*x*pi/5) + 1.52*sin(2*x*pi/5)
## Make sure that you really got it!!!!
### ---- ### 

## Let's now regard pol as a function as p(1) will throw an no method error!!!!
function (p::harmonic_pol)(x::Real)
    ## Your code here!!!!
    return nothing
end
### let's give a check with polynomial that you create above
pol(0), 1.0768932910993034
### Are these number the same???, if so run the following unit_test 

function unit_test_1()
    @assert(student_number != 0,"write your student number")
    l::Int64 = 0
    @showprogress for i in 1:100
        sleep(0.001)
        begin
        Random.seed!(i)
        pol = harmonic_pol(5, randn(3))
        res = isapprox(pol(0), sum(pol.coeff) - pol.coeff[3])
        res_ = isapprox(pol(5/4), sum(pol.coeff) - pol.coeff[2])
        l += res && res_ ? 1 : 0        
        end
    end
    if l == 100
        @info "Congrats comrade!!! you got it!!! You Finally GOT it!!!! You can celebrate it now!!!!"
    else
        throw(AssertionError("Yaa doin' wrong buddy!!!"))
    end
    return 1
end
### Run the unit test, and grab the points!!!! ###
unit_test_1()
### end of unit_test_1 ####



### Below I partially implement fit! function the rest is your job!!!
function fit!(pol::harmonic_pol, x::Vector{Float64}, y::Vector{Float64})
    size_ = x |> size
    X_feature::Matrix{Float64} = ones(size_[1],3)
    X_feature[:, 2] = cos.(2*pi*x/pol.period) # 1
    X_feature[:, 3] = sin.(2*pi*x/pol.period) # 2

    #= X_feature matrix is of the following format
    # |1 * *| 
    # |1 * *|
    # |1 * *|
    # |1 * *|
    # each star comes from cos and sin as decribe in #1 and #2
    # Below you should come up with a vector V so that
    # ||X_feature*V - y||^2 is minimal  
    # DO NOT USE "\" operation!!!
    # ---
    # your code here
    # ---    
    # you will then be done by setting: =#
end

let
    ## Run the following for debugging 2 see ha ya doin'
    begin
        coeff, period = randn(3), rand(DiscreteUniform(10, 30))
    end
        x = collect(0:0.1:100)
        pol = harmonic_pol(period, coeff)
        y = map(pol, x)
        pol_ = harmonic_pol(period, randn(3))
        fit!(pol_, x, y)
        isapprox(coeff, pol_.coeff; rtol = 1e-1) ? 1 : 0
end



### If you think that things are good
### run the following tests and enjoy the rest of the day
function unit_test_2()
    l::Int64 = 0
    x = collect(0:0.1:100)
    @showprogress for q in 1:250
        sleep(0.1)
        coeff, period = randn(3), rand(DiscreteUniform(10, 30))
        pol = harmonic_pol(period, coeff)
        y = pol.(x)
        pol_ = harmonic_pol(period, randn(3))
        fit!(pol_, x, y)
        l += 1*isapprox(coeff, pol_.coeff)
        end
    if l == 250
        println("Good!!! you doin' good!!! passed!!!")
        return 1
    else
        throw(
            "Something is WRONG!!!!"
        )
    end
end

### run unit_test_2 ###
unit_test_2()
### end of unit_test_2 ###


###### do not change anything below!!!!
##### No need to run anything below!!!!
###### As this may cause compiler to crash, and degradation!!!

if abspath(PROGRAM_FILE) == @__FILE__
    G::Int64 = unit_test_1()  + unit_test_2()
    dict_ = Dict("std_ID"=>student_number, "G"=>G)
    try
        BSON.@save "$(student_number).res" dict_ 
        catch Exception 
            println("something went wrong with", Exception)
    end

end

