module Gauss

using Distributions, StaticArrays

struct Gaussian{T,S}
    μ::T
    ΣU::S
end

function Gaussian(X::MvNormal)
    n = length(mean(X))
    Gaussian(SVector{n}(mean(X)), chol(SMatrix{n,n}(cov(X))))
end

Base.cov(X::Gaussian) = X.ΣU'X.ΣU

@inline function Base.rand(X::Gaussian{T}) where {T<:SVector{N}} where N
    ε = @SVector(randn(N))
    return X.ΣU'*ε
end

end