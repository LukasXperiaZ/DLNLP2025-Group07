How to install
===

1. install poetry this [WEBSITE](https://python-poetry.org/docs/#installing-with-the-official-installer) is the official install guide.
2. Configure that poetry will create the .venv folder in the project directory (VSCode will have it easier that way to recognize it): `poetry config virtualenvs.in-project true`
3. run `poetry install --no-root` to install the neccessary dependencies
4. Open e.g. a jupyter notebook and set the python interpreter to be the one in the .venv directory