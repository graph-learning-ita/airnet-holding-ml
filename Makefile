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
	cd docs/tcc && latexmk -shell-escape -interaction=nonstopmode -f -pdf -file-line-error 1_main.tex || true
	make clean

article:
	make clean
	cd docs/article && TEXINPUTS=".:tcc:" latexmk -shell-escape -interaction=nonstopmode -f -pdf -file-line-error article.tex || true
	make clean

article-watch:
	make clean
	cd docs/article && TEXINPUTS=".:tcc:" latexmk -pvc -shell-escape -interaction=nonstopmode -f -pdf -file-line-error article.tex
	make clean

tcc-presentation:
	make clean
	cd docs/tcc-presentation && latexmk -interaction=nonstopmode -f -shell-escape -pdf -file-line-error apresentacao.tex
	make clean

seminary:
	make clean
	cd docs/seminary && latexmk -shell-escape -interaction=nonstopmode -f -pdf -file-line-error apresentacao.tex || true
	make clean

clean:
	@echo -n "Limpando arquivos auxiliares...\n"
	cd docs && find . -type f \( -name "*.out" -o -name "*.aux" -o -name "*.alg" -o -name "*.acr" -o -name "*.dvi"  \
				-o -name "*.log" -o -name "*.bbl" -o -name "*.blg" -o -name "*.ntn" -o -name "*.not" -o -name "*.lof" \
				-o -name "*.lot" -o -name "*.toc" -o -name "*.loa" -o -name "*.lsg" -o -name "*.nlo" -o -name "*.nls" \
				-o -name "*.ilg" -o -name "*.lol" -o -name "*.ind" -o -name "*.ist" -o -name "*.glg" -o -name "*.glo" \
				-o -name "*.xdy" -o -name "*.acn" -o -name "*.idx" -o -name "*.loq" -o -name "*~" -o -name "*.gls" \
				-o -name "*.bcf" -o -name "*.nav" -o -name "*.run.xml" -o -name "*.snm" -o -name "*.fdb_latexmk" -o -name "*.fls" \) \
                | xargs rm -vf
	@echo "Processo finalizado com sucesso!"


.PHONY: build run pull check lint coverage notebook install
