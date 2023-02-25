struct GradientQLearning
    𝒜 # action space (assumes 1:nactions)
    γ # discount
    Q # parameterized action value function Q(θ,s,a) ∇Q # gradient of action value function
    θ # action value function parameter
    α # learning rate
end
function lookahead(model::GradientQLearning, s, a)
    return model.Q(model.θ, s,a)
end 

function update!(model::GradientQLearning, s, a, r, s′)
    𝒜, γ, Q, θ, α = model.𝒜, model.γ, model.Q, model.θ, model.α u = maximum(Q(θ,s′,a′) for a′ in 𝒜)
    Δ = (r + γ*u - Q(θ,s,a))*model.∇Q(θ,s,a)
    θ[:] += α*scale_gradient(Δ, 1)
    return model
end 



β(s,a) = [s,s^2,a,a^2,1]
Q(θ,s,a) = dot(θ,β(s,a))
∇Q(θ,s,a) = β(s,a)
θ = [0.1,0.2,0.3,0.4,0.5] # initial parameter vector α = 0.5 # learning rate
model = GradientQLearning(𝒫.𝒜, 𝒫.γ, Q, ∇Q, θ, α) ε = 0.1 # probability of random action
π = EpsilonGreedyExploration(ε)
k = 20 # number of steps to simulate
s = 0.0 # initial state simulate(𝒫, model, π, k, s)



mutable struct LocallyWeightedValueFunction 
    k # kernel function k(s, s′)
    S # set of discrete states
    θ # vector of values at states in S
end 
function (Uθ::LocallyWeightedValueFunction)(s)
    w = normalize([Uθ.k(s,s′) for s′ in Uθ.S], 1)
    return Uθ.θ ⋅ w
end
function fit!(Uθ::LocallyWeightedValueFunction, S, U) 
    Uθ.θ = U
    return Uθ
end 