pipeline {
    agent any

    environment {
        APP_NAME = 'meow-cdd-ai'
    }

    stages {
        stage('Checkout') {
            steps {
                echo 'Checking out meowCDDAI source code...'
                checkout scm
            }
        }

        stage('Docker Build') {
            steps {
                echo 'Building Docker image...'
                sh "docker build -t ${APP_NAME}:latest ."
            }
        }

        stage('Deploy') {
            steps {
                echo 'Deploying meow-cdd-ai via docker compose...'
                sh '''
                    docker compose -p meow \
                        -f /var/jenkins_home/workspace/meow-compose/docker-compose.yml \
                        up -d --no-deps --force-recreate meow-cdd-ai
                '''
            }
        }
    }

    post {
        success {
            echo "✅ ${APP_NAME} pipeline completed successfully!"
        }
        failure {
            echo "❌ ${APP_NAME} pipeline failed! Check console logs."
        }
        always {
            cleanWs()
        }
    }
}
