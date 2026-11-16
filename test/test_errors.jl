# The exception hierarchy: every exception type this package defines subtypes the root
# `CellularAutomataInferenceError` and shows the offending names.
function owned_exception_types(M)
    types = Type[]
    for n in names(M; all=true)
        isdefined(M, n) || continue
        T = getglobal(M, n)
        T isa Type && T <: Exception && parentmodule(T) === M && push!(types, T)
    end
    return types
end

@testset "errors" begin
    owned = owned_exception_types(CellularAutomataInference)
    @test CellularAutomataInferenceError in owned
    @test isabstracttype(CellularAutomataInferenceError)
    for T in owned
        @test T <: CellularAutomataInferenceError
    end
    e = SupportError("β", -1.0, "outside the prior")
    @test sprint(showerror, e) == "SupportError: β = -1.0: outside the prior"
end
