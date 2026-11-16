using Documenter
using DocumenterCitations
using CellularAutomataInference

DocMeta.setdocmeta!(CellularAutomataInference, :DocTestSetup,
                    :(using CellularAutomataInference);
                    recursive=true)

pages = Any["Home" => "index.md",
            "API Reference" => "api.md",
            "References" => "references.md"]

# Source links need a commit to point at. A checkout without commits (a fresh
# scaffold, a tarball) builds without them instead of failing.
has_commit = success(pipeline(`git -C $(@__DIR__) rev-parse HEAD`; stderr=devnull))
remote_kw = has_commit ?
            (; repo=Remotes.GitHub("ecorecipes", "CellularAutomataInference.jl")) :
            (; remotes=nothing)

# Bibliography for `[Key](@cite)` citations in docstrings and pages. The .bib is
# synced from the workspace-level docs/references.bib by scripts/sync_references.jl.
bib = CitationBibliography(joinpath(@__DIR__, "src", "references.bib");
                           style=:authoryear)

makedocs(;
         remote_kw...,
         modules=[CellularAutomataInference],
         sitename="CellularAutomataInference.jl",
         authors="Simon Frost",
         format=Documenter.HTML(;
                                prettyurls=get(ENV, "CI", "false") == "true",
                                canonical="https://ecorecipes.github.io/CellularAutomataInference.jl",
                                repolink="https://github.com/ecorecipes/CellularAutomataInference.jl",
                                edit_link="main"),
         pages=pages,
         plugins=[bib])

"--no-deploy" in ARGS || deploydocs(;
                                    repo="github.com/ecorecipes/CellularAutomataInference.jl.git",
                                    devbranch="main")
