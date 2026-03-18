setup:
	python3 -m venv .venv
	. .venv/bin/activate && pip install -e .
	docker compose up -d
	ollama pull nomic-embed-text

run:
	. .venv/bin/activate && atenea-server

clean-index:
	curl -X DELETE http://localhost:6333/collections/atenea_code

clean:
	docker compose down
	rm -rf .venv
