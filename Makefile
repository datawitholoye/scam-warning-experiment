.PHONY: install simulate analyse test lint

install:        ## install everything into the active virtual environment
	pip install -r requirements-dev.txt && pip install -e .

simulate:       ## generate data/ and reports/simulation_summary.md (~15 s)
	python -m scam_experiment.simulate

test:
	pytest -q

lint:
	ruff check .

analyse:        ## Day 2: A/B readout -> reports/day2_ab_test.md (~10 s)
	python -m scam_experiment.analysis
