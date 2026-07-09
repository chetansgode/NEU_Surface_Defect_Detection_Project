@echo off
echo Stopping Docker containers...
docker-compose -f docker-compose.yml -p project-4 down

pause