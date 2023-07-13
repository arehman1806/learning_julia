using Pkg
# Pkg.add("FIB")

using POMDPs, POMDPModelTools
using POMDPSimulators
using POMDPPolicies
using FIB # For the solver

struct TigerPOMDP <: POMDP{Bool, Symbol, Bool}
    r_listen::Float64 # reward for listening (default -1)
    r_findtiger::Float64 # reward for finding the tiger (default -100)
    r_escapetiger::Float64 # reward for escaping (default 10)
    p_listen_correctly::Float64 # prob of correctly listening (default 0.85)
    discount_factor::Float64 # discount
end

TigerPOMDP() = TigerPOMDP(-1., -100., 10., 0.85, 0.95)

POMDPs.states(pomdb::TigerPOMDP) = [true, false]
POMDPs.stateindex(pomdp::TigerPOMDP, s::Bool) = s ? 1 : 2 ;

POMDPs.actions(pomdp::TigerPOMDP) = [:open_left, :open_right, :listen]
function POMDPs.actionindex(pomdb::TigerPOMDP, a::Symbol)
    if a == :open_right
        return 2
    elseif a == :open_left
        return 1
    elseif a==:listen
        return 3
    end
    error("invalid TigerPOMDP action: $a")
end;



function POMDPs.transition(pomdp::TigerPOMDP, s::Bool, a::Symbol)
    if a == :open_left || a == :open_right
        # problem resets
        return BoolDistribution(0.5) 
    elseif s
        # tiger on the left stays on the left 
        return BoolDistribution(1.0)
    else
        return BoolDistribution(0.0)
    end
end
     
POMDPs.observations(pomdp::TigerPOMDP) = [true, false]
POMDPs.obsindex(pomdp::TigerPOMDP, o::Bool) = o+1

function POMDPs.observation(pomdp::TigerPOMDP, a::Symbol, s::Bool)
    pc = pomdp.p_listen_correctly
    if a == :listen 
        if s 
            return BoolDistribution(pc)
        else
            return BoolDistribution(1 - pc)
        end
    else
        return BoolDistribution(0.5)
    end
end

using POMDPSimulators
using POMDPPolicies

m = TigerPOMDP()

# policy that takes a random action
policy = RandomPolicy(m)

for (s, a, r) in stepthrough(m, policy, "s,a,r", max_steps=10)
    @show s
    @show a
    @show r
    println()
end
     

     
# reward model
function POMDPs.reward(pomdp::TigerPOMDP, s::Bool, a::Symbol)
    r = 0.0
    if a == :listen
        r+=pomdp.r_listen
    elseif a == :open_left
        s ? (r += pomdp.r_findtiger) : (r += pomdp.r_escapetiger)
    elseif a == :open_right
        s ? (r += pomdp.r_escapetiger) : (r += pomdp.r_findtiger)
    end
    return r
end


POMDPs.initialstate_distribution(pomdp::TigerPOMDP) = BoolDistribution(0.5)

POMDPs.discount(pomdp::TigerPOMDP) = pomdp.discount_factor



using POMDPSimulators
using POMDPPolicies

m = TigerPOMDP()

# policy that takes a random action
rand_policy = RandomPolicy(m)

for (s, a, r) in stepthrough(m, rand_policy, "s,a,r", max_steps=10)
    @show s
    @show a
    @show r
    println()
end

solver = FIBSolver(max_iterations=100000000000, tolerance=0.000000001)
fib_policy = solve(solver, m)
rollout_sim = RolloutSimulator(max_steps=10)
fib_reward = simulate(rollout_sim, m, fib_policy);
rand_reward = simulate(rollout_sim, m, rand_policy);


print(fib_reward)
println()
print(rand_reward)
     



     
