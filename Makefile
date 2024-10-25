PROJECT_NAME = dsc2024
DOCKER_REGISTRY := ryukinix/dsc2024
VERSION := latest
UID = $(shell id -u)
GID = $(shell id -g)
DOCKER = docker
ifeq ($(DOCKER), podman)
	DOCKER_FLAGS=--userns keep-id
else
	DOCKER_FLAGS=
endif

DOCKER_RUN = $(DOCKER) run $(DOCKER_FLAGS) \
					--user $(UID):$(GID) \
					-e HOME=/tmp --rm \
					-t \
					-v $(PWD)/tests:/app/tests \
					-w /app
MOUNT_NOTEBOOK = -v $(PWD)/notebooks:/app/notebooks -v $(PWD)/dsc2024:/app/dsc2024 -v $(PWD)/datasets:/app/datasets
EXPOSE_PORT = --net=host
MESSAGE ?= submitted through makefile
DATASET_SUBMIT_PATH ?= datasets/catboost_submit.csv

install: # install locally
	python -m venv .venv
	source .venv/bin/activate
	pip install -U pdm setuptools wheel
	pdm install

run: build
	$(DOCKER_RUN) $(PROJECT_NAME)

pull:
	$(DOCKER) pull $(DOCKER_REGISTRY)

build:
	$(DOCKER) build -t $(PROJECT_NAME) .

publish: build
	$(DOCKER) tag $(PROJECT_NAME) $(DOCKER_REGISTRY):$(VERSION)
	$(DOCKER) push $(DOCKER_REGISTRY):$(VERSION)

check: build
	$(DOCKER_RUN) $(PROJECT_NAME) check
	sed -i "s|/app|$(PWD)|g" tests/coverage.xml

submit:
	kaggle competitions submit -c data-science-challenge-at-eef-2024 -f "$(DATASET_SUBMIT_PATH)" -m "$(MESSAGE)"

lint: build
	$(DOCKER_RUN) $(PROJECT_NAME) lint dsc2024/ tests/

notebook: build
	$(DOCKER_RUN) -i $(MOUNT_NOTEBOOK) $(EXPOSE_PORT) $(PROJECT_NAME) jupyter lab --allow-root

coverage:
	coverage html
	open htmlcov/index.html

tcc:
	make clean
	cd docs/tcc && latexmk -shell-escape -pdf -file-line-error 1_main.tex
	make clean

tcc-presentation:
	make clean
	cd docs/tcc-presentation && latexmk -shell-escape -pdf -file-line-error apresentacao.tex
	make clean

seminary:
	make clean
	cd docs/seminary && latexmk -shell-escape -pdf -file-line-error apresentacao.tex
	make clean

clean:
	@echo -n "Limpando arquivos auxiliares...\n"
	@cd docs/tcc && rm -v -f *.out *.aux *.alg *.acr *.dvi *.gls \
		*.log *.bbl *.blg *.ntn *.not *.lof \
		*.lot *.toc *.loa *.lsg *.nlo *.nls \
		*.ilg *.lol *.ind *.ist *.glg *.glo *.xdy *.acn *.idx *.loq *~ \
		*.bcf *.nav *.run.xml *.snm *.fdb_latexmk *.fls
	@cd docs/seminary && rm -v -f *.out *.aux *.alg *.acr *.dvi *.gls \
		*.log *.bbl *.blg *.ntn *.not *.lof \
		*.lot *.toc *.loa *.lsg *.nlo *.nls \
		*.ilg *.lol *.ind *.ist *.glg *.glo *.xdy *.acn *.idx *.loq *~ \
		*.bcf *.nav *.run.xml *.snm *.fdb_latexmk *.fls
	@cd docs/tcc-presentation && rm -v -f *.out *.aux *.alg *.acr *.dvi *.gls \
		*.log *.bbl *.blg *.ntn *.not *.lof \
		*.lot *.toc *.loa *.lsg *.nlo *.nls \
		*.ilg *.lol *.ind *.ist *.glg *.glo *.xdy *.acn *.idx *.loq *~ \
		*.bcf *.nav *.run.xml *.snm *.fdb_latexmk *.fls
	@echo "Processo finalizado com sucesso!"


.PHONY: build run pull check lint coverage notebook install
