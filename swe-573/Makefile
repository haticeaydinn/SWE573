build:
	docker build --force-rm -t swe-573-website:latest .

compose-start:
	docker-compose up --remove-orphans

compose-stop:
	docker-compose down --remove-orphans

compose-manage-py-migration:
	docker-compose run --rm website python manage.py makemigrations