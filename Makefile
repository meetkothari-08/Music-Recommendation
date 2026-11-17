.PHONY: install data evaluate app test lint notebook djset

install:      ## install the package with dev + app extras
	pip install -e ".[dev,notebook]"

data:         ## download and clean the dataset -> data/processed/tracks.parquet
	music-rec build

evaluate:     ## run offline evaluation -> reports/
	music-rec evaluate

app:          ## launch the Streamlit app
	streamlit run app/streamlit_app.py

test:
	pytest -q

lint:
	ruff check . && ruff format --check .

notebook:     ## re-execute the EDA notebook in place
	jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb

djset:        ## example DJ set from the command line
	python -m music_recommender djset "chill house set that builds into a party, 10 songs"
