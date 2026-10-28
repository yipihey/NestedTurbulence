.PHONY: test data check brand site preview all

# the validation suite of the Julia core (cited by the ✓ badges)
test:
	julia --project=core -e 'using Pkg; Pkg.instantiate(); Pkg.test()'

# every dataset of the book and the laboratory (app/data/*.json); NAMES="ghost convergence" for a subset
data:
	julia -t auto --project=core core/bench/run_experiments.jl $(NAMES)

# every provenance link of the text resolves
check:
	python3 scripts/check_links.py

brand:
	python3 scripts/brand_to_css.py

site:
	quarto render

preview:
	quarto preview

all: test check site
