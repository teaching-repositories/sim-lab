.PHONY: requirements

requirements:
	poetry export -f requirements.txt --output requirements.txt --without-hashes
	poetry export --dev -f requirements.txt --output requirements-dev.txt --without-hashes
	cat requirements-dev.txt >> requirements.txt
	rm requirements-dev.txt

generate-setup:
	dephell deps convert --from=pyproject.toml --to=setup.py
