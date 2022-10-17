/*******************************************************************************
 * Copyright (c) 2021-2022, National Research Foundation (SARAO)
 *
 * Licensed under the BSD 3-Clause License (the "License"); you may not use
 * this file except in compliance with the License. You may obtain a copy
 * of the License at
 *
 *   https://opensource.org/licenses/BSD-3-Clause
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ******************************************************************************/

/* This file tells Jenkins how to test this repo. Everything runs in docker
 * containers, so in theory any Jenkins server should be able to parse this
 * file, although the Jenkins server needs access to a GPU with tensor cores.
 * This means that the Docker Engine being used needs to have been configured
 * to use the Nvidia Container Runtime and the node which the container runs
 * on needs a Nvidia GPU with tensor cores along with the Nvidia Driver installed.
 *
 * Additionally the Jenkins server also needs access to a Mellanox NIC that
 * supports ibverbs. The ibverbs drivers need to be passed into the Jenkins
 * container using specific flags.
 */

pipeline {
  agent {
    dockerfile {
      registryCredentialsId 'dockerhub'  // Supply credentials to avoid rate limit

      /* Use the Jenkins-specific stage of the Dockerfile as the image for
       * testing. This provides the appropriate dependencies.
       */
      additionalBuildArgs '--target=jenkins'

      /* The following argument needs to be specified in order for the container
       * to launch correctly.
       *
       * --gpus=all: This argument passes the Nvidia driver and devices from the
       * host to the container. It requires the NVIDIA Container Runtime to be
       * installed on the host.
       *
       * -v /var/run/docker.sock:/var/run/docker.sock: makes the connection to
       * the Docker server available inside the container (for building a Docker
       * image).
       */
      args '--gpus=all -v /var/run/docker.sock:/var/run/docker.sock'
    }
  }

  options {
    timeout(time: 2, unit: 'HOURS')
  }

  stages {
    stage('Install Python packages') {
      steps {
        sh 'pip install -r requirements.txt -r requirements-dev.txt'
      }
    }

    stage('Install katgpucbf package') {
      steps {
        sh 'pip install --no-deps ".[test]" && pip check'
      }
    }

    stage('Parallel stage') {
      parallel {
        stage('Compile C++ tools') {
          steps {
            // Make and compile tools.
            dir('src/tools') {
              sh 'make clean'
              sh 'make -j'
            }
          }
        }

        /* This stage ensures that all the python style guidelines checks pass.
         * This will catch if someone has committed to the repo without
         * installing the required pre-commit hooks, or has bypassed them.
         */
        stage('Run pre-commit checks') {
          steps {
            sh 'pre-commit install'
            // no-commit-to-branch complains if we are on the main branch
            sh 'SKIP=no-commit-to-branch pre-commit run --all-files'
          }
        }

        /* This stage actually runs pytest. Pytest has a number of flags that are
         * not required but make life easier:
         * 1. -n X: Launches X threads and runs the tests in parallel across
         *     multiple threads. This speeds up testing significantly. NOTE: This
         *     can create resource contention over things like GPU RAM. If it
         *     starts becoming an issue set X to 1. I have noticed an issue once
         *     where sometimes one thread got stuck and it stalled the pipeline.
         *     Until this has been solved, I am removing this argument entirely
         * 2. -v: Increases verbosity
         * 3. --junitxml=reports/result.xml' Writes the results to a file for later
         *    examination.
         */
        stage('Run pytest (quick)') {
          when { not { anyOf { changeRequest target: 'main'; branch 'main' } } }
          options { timeout(time: 10, unit: 'MINUTES') }
          steps {
            sh 'pytest -v -ra --junitxml=reports/result.xml --cov=katgpucbf --cov=test --cov-report=xml --cov-branch'
          }
        }
        stage('Run pytest (full)') {
          when { anyOf { changeRequest target: 'main'; branch 'main' } }
          options { timeout(time: 60, unit: 'MINUTES') }
          steps {
            sh 'pytest -v -ra --all-combinations --junitxml=reports/result.xml --cov=test --cov=katgpucbf --cov-report=xml --cov-branch'
          }
        }

        stage('Build documentation') {
          options { timeout(time: 5, unit: 'MINUTES') }
          steps {
            // -W causes warnings to become errors.
            // --keep-going ensures we get all warnings instead of just the first.
            sh 'make -C doc clean html latexpdf SPHINXOPTS="-W --keep-going"'
            publishHTML(target: [reportName: 'Module documentation', reportDir: 'doc/_build/html', reportFiles: 'index.html'])
            publishHTML(target: [reportName: 'Module documentation (PDF)', reportDir: 'doc/_build/latex', reportFiles: 'katgpucbf.pdf'])
          }
        }
      }
    }

    stage('Publish test results') {
      steps {
        junit 'reports/result.xml'
        cobertura coberturaReportFile: 'coverage.xml'
      }
    }

    stage('Build and push Docker image') {
      when { not { changeRequest() }}
      environment {
        DOCKER_BUILDKIT = '1'
      }
      steps {
        script {
          branch = env.BRANCH_NAME
          tag = (branch == "main") ? "latest" : branch
          // Supply credentials to Dockerhub so that we can reliably pull the base image
          docker.withRegistry("https://docker.io/", "dockerhub") {
            dockerImage = docker.build(
              "harbor.sdp.kat.ac.za/cbf/katgpucbf:${tag}",
              "--pull "
              + "--label=org.opencontainers.image.revision=${env.GIT_COMMIT} "
              + "--label=org.opencontainers.image.source=${env.GIT_URL} ."
            )
          }
          docker.withRegistry("https://harbor.sdp.kat.ac.za/", "harbor-cbf") {
            dockerImage.push()
          }
        }
      }
    }
  }

  /* This post stage is configured to always run at the end of the pipeline,
   * regardless of the completion status. In this stage an email is sent to
   * the specified address with details of the Jenkins job and an XML file
   * containing the pytest results. The final step removes the workspace when
   * the build is complete.
   */
  post {
    always {
      emailext attachLog: true,
      attachmentsPattern: 'reports/result.xml',
      body: """<b>Overall Test Results:</b> ${env.JOB_NAME} - Build#${env.BUILD_NUMBER} - ${currentBuild.result}<br>
      <b>Node:</b> ${env.NODE_NAME}<br>
      <b>Duration:</b> ${currentBuild.durationString}<br>
      <b>Build URL:</b> ${env.BUILD_URL}<br>
      <br>
      <i>Note: This is an Automated email notification.</i>""",
      recipientProviders: [developers(), requestor(), culprits()],
      subject: '$PROJECT_NAME - $BUILD_STATUS!',
      to: '$DEFAULT_RECIPIENTS'


      cleanWs()
    }
  }

}
