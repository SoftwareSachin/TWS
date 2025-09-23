#!/usr/bin/make

include .env

define SERVERS_JSON
{
	"Servers": {
		"1": {
			"Name": "fastapi-alembic",
			"Group": "Servers",
			"Host": "$(DATABASE_HOST)",
			"Port": 5432,
			"MaintenanceDB": "postgres",
			"Username": "$(DATABASE_USER)",
			"SSLMode": "prefer",
			"PassFile": "/tmp/pgpassfile"
		}
	}
}
endef
export SERVERS_JSON

help:
	@echo "make"
	@echo "    install"
	@echo "        Install all packages of poetry project locally."
	@echo "    run-dev-build"
	@echo "        Run development docker compose and force build containers."
	@echo "    run-dev"
	@echo "        Run development docker compose."
	@echo "    stop-dev"
	@echo "        Stop development docker compose."
	@echo "    run-prod"
	@echo "        Run production docker compose."
	@echo "    stop-prod"
	@echo "        Run production docker compose."
	@echo "    init-db"
	@echo "        Init database with sample data."	
	@echo "    add-dev-migration"
	@echo "        Add new database migration using alembic."
	@echo "    upgrade-migration"
	@echo "        This helps to upgrade pending migrations."	
	@echo "    run-pgadmin"
	@echo "        Run pgadmin4."	
	@echo "    load-server-pgadmin"
	@echo "        Load server on pgadmin4."
	@echo "    clean-pgadmin"
	@echo "        Clean pgadmin4 data."
	@echo "    formatter"
	@echo "        Apply black formatting to code."
	@echo "    lint"
	@echo "        Lint code with ruff, and check if black formatter should be applied."
	@echo "    lint-watch"
	@echo "        Lint code with ruff in watch mode."
	@echo "    lint-fix"
	@echo "        Lint code with ruff and try to fix."	
	@echo "    run-sonarqube"
	@echo "        Starts Sonarqube container."
	@echo "    run-sonar-scanner"
	@echo "        Starts Sonarqube container."	
	@echo "    stop-sonarqube"
	@echo "        Stops Sonarqube container."
	@echo "    run-test-video"
	@echo "        Run test environment with video ingestion support."
	@echo "    run-test-video-build"
	@echo "        Build and run test environment with video ingestion support."
	@echo "    pytest-video"
	@echo "        Run all video ingestion tests (CPU-only with mocked dependencies)."
	@echo "    pytest-video-unit"
	@echo "        Run video ingestion unit tests."
	@echo "    pytest-video-integration"
	@echo "        Run video ingestion integration tests."
	@echo "    pytest-video-api"
	@echo "        Run video ingestion API tests."
	@echo "    pytest-video-coverage"
	@echo "        Run video ingestion tests with coverage reporting."

install:
	cd backend/app && \
	poetry shell && \
	poetry install

run-dev-build:
	docker compose -f docker-compose-dev.yml up --build

run-dev-build-video:
	docker compose -f docker-compose-dev.yml --profile video-ingestion up --build 

images-k8s-build:
	docker compose -f docker-compose-for-k8s.yml build

run-dev:
	docker compose -f docker-compose-dev.yml up

run-dev-video:
	docker compose -f docker-compose-dev.yml --profile video-ingestion up

stop-dev:
	docker compose -f docker-compose-dev.yml down

run-prod:
	docker compose up

stop-prod:
	docker compose down

create-celery-db:
	if ! docker compose -f docker-compose-dev.yml exec database psql -U ${DATABASE_USER} -h localhost -lqt | cut -d \| -f 1 | grep -qw ${DATABASE_CELERY_NAME}; then \
		docker compose -f docker-compose-dev.yml exec database createdb -U ${DATABASE_USER} -W ${DATABASE_PASSWORD} -h localhost -O ${DATABASE_USER} ${DATABASE_CELERY_NAME}; \
	fi

init-db:
	docker compose -f docker-compose-dev.yml exec fastapi-server python app/initial_data.py && \
	echo "Initial data created." 

formatter:
	cd backend/app && \
	poetry run isort app test && \
	poetry run black app test

lint:
	cd backend/app && \
	poetry run ruff check app && poetry run black --check app

bandit:
	cd backend/app && \
	poetry run bandit -r . -c pyproject.toml

mypy:
	cd backend/app && \
	poetry run mypy .

lint-watch:
	cd backend/app && \
	poetry run ruff check app --watch

lint-fix:
	cd backend/app && \
	poetry run ruff check app --fix

run-sonarqube:
	docker compose -f docker-compose-sonarqube.yml up

stop-sonarqube:
	docker compose -f docker-compose-sonarqube.yml down

run-sonar-scanner:
	docker run --rm -v "${PWD}/backend:/usr/src" sonarsource/sonar-scanner-cli -X

add-dev-migration:
	docker compose -f docker-compose-dev.yml exec fastapi-server alembic revision --autogenerate && \
	docker compose -f docker-compose-dev.yml exec fastapi-server alembic upgrade head && \
	echo "Migration added and applied."

upgrade-migration:
	docker compose -f docker-compose-dev.yml exec fastapi-server alembic upgrade head && \
	echo "Migration upgraded."

merge-heads-migration:
	docker compose -f docker-compose-dev.yml exec fastapi-server alembic merge heads && \
	docker compose -f docker-compose-dev.yml exec fastapi-server alembic upgrade head && \
	echo "Merge Migration added and applied."

run-pgadmin:
	echo "$$SERVERS_JSON" > ./pgadmin/servers.json && \
	docker volume create pgadmin_data && \
	docker compose -f pgadmin.yml up --force-recreate

clean-pgadmin:
	docker volume rm pgadmin_data

run-radon-cc:
	cd backend/app && \
	poetry run radon cc . -nc -e "test/*, load-test/*" -s --total-average

run-test:
	docker compose -f docker-compose-test.yml up

run-test-build:
	docker compose -f docker-compose-test.yml up --build

pytest-cov:
	docker compose -f docker-compose-test.yml exec fastapi-server pytest --cov=app 

pytest-cov-xml:
	docker compose -f docker-compose-test.yml exec fastapi-server pytest --cov=app --cov-report xml:.coverage.xml

pytest-cov-html:
	docker compose -f docker-compose-test.yml exec fastapi-server pytest --cov=app --cov-report=html

pytest-full:
	docker compose -f docker-compose-test.yml exec fastapi-server pytest

pytest:
	docker compose -f docker-compose-test.yml exec fastapi-server pytest test/api/ test/api/v2/ test/unit_test/ test/integration_tests/ --ignore=test/api/test_search.py --disable-warnings

pytest-unit:
	docker compose -f docker-compose-test.yml exec fastapi-server pytest test/unit_test/

pytest-api:
	docker compose -f docker-compose-test.yml exec fastapi-server pytest test/api/

pytest-integration:
	docker compose -f docker-compose-test.yml exec fastapi-server pytest test/integration_tests/

pytest-v2:
	docker compose -f docker-compose-test.yml exec fastapi-server pytest test/api/v2/

create-k8s-kompose-yaml:
	kompose --file docker-compose-for-k8s.yml convert --out k8s/kompose/

remove-k8s-kompose-yaml:
	rmdir .\k8s\kompose /s /q && \
	mkdir .\k8s\kompose

run-k8s-amplifi-local:
	kubectl apply -f .\k8s\local -R --namespace amplifi-dev

stop-k8s-amplifi-local:
	kubectl delete pods,services --all -n amplifi-dev

delete-k8s-amplifi-local:
	kubectl delete all --all -n amplifi-dev