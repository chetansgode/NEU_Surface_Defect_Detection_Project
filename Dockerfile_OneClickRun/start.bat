@echo off
echo Starting Docker containers...

docker compose -f docker-compose.yml -p project-4 up -d

timeout /t 5

start http://localhost:8501
start http://localhost:8000/docs

pause


