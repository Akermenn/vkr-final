pipeline {
    agent any

    stages {
        stage('Cleanup') {
            steps {
                // Удаляем старое, если есть. || true не дает упасть ошибке.
                sh 'docker stop project-frontend project-backend test-runner || true'
                sh 'docker rm project-frontend project-backend test-runner || true'
                sh 'docker network create app-network || true'
            }
        }

        stage('Build & Run App') {
            steps {
                // 1. Бэкенд
                sh 'docker build -t my-ds-backend ./backend'
                // Запускаем и пробрасываем порт 5000 наружу
                sh 'docker run -d --name project-backend --network app-network -p 5000:5000 my-ds-backend'

                // 2. Фронтенд
                sh 'docker build -t my-ds-frontend ./frontend'
                sh 'docker run -d --name project-frontend --network app-network -p 80:80 my-ds-frontend'

                // 3. Пауза 20 секунд, пока модели обучаются
                echo "Waiting for models to train..."
                sh 'sleep 20'
            }
        }

        stage('Test (Playwright)') {
            steps {
                // Собираем тестировщика
                sh 'docker build -f Dockerfile.test -t my-test-runner .'

                // Запускаем тесты в той же сети
                sh 'docker run --rm --network app-network --name test-runner my-test-runner pytest e2e_tests/test_frontend.py'
            }
        }
    }
}