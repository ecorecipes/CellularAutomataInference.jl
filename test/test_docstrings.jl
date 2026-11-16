# Every exported name this package owns has a docstring. A name re-exported from another
# package (AbstractMCMC's `sample`, say) is that package's to document, so it is skipped.
function owned_undocumented(M)
    return filter(n -> Base.which(M, n) === M, Base.Docs.undocumented_names(M))
end

@testset "docstrings on every exported name" begin
    @test isempty(owned_undocumented(CellularAutomataInference))
end
