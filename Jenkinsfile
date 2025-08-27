/*******************************************************************************
 * Copyright (c) 2021-2025, National Research Foundation (SARAO)
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
 */

pipeline {
  agent { label 'katgpucbf' }
  options {
    timeout(time: 2, unit: 'HOURS')
    disableConcurrentBuilds()
  }

  stages {
    /* This is an outer stage that serves just to hold the 'agent' config for
     * stages that run inside a Docker container. The build of the Docker
     * image is done outside of that Docker container because of the
     * complexities of running Docker-in-docker.
     */
    stage('Testing') {
      agent {
        dockerfile {
          reuseNode true
          registryCredentialsId 'dockerhub'  // Supply credentials to avoid rate limit

          /* Use the Jenkins-specific stage of the Dockerfile as the image for
           * testing. This provides the appropriate dependencies.
           */
          additionalBuildArgs '--target=jenkins'

          /* The following arguments needs to be specified in order for the container
           * to launch correctly.
           *
           * --runtime=nvidia --gpus=all: This argument passes the NVIDIA driver and
           * devices from the host to the container. It requires the NVIDIA Container
           * Toolkit to be installed on the host.
           */
          args '--runtime=nvidia --gpus=all'
        }
      }
      stages {
        stage('Install katgpucbf package') {
          steps {
            sh 'pip install --no-deps ".[test]" && pip check'
          }
        }

        stage('Parallel stage') {
          parallel {
            stage('Compile and test microbenchmarks') {
              options { timeout(time: 5, unit: 'MINUTES') }
              steps {
                dir('scratch') {
                  sh 'make clean'
                  sh 'make'
                  sh './memcpy_loop -T'

                  // We just want to know if they run without crashing, so we use a small
                  // number of passes to speed things up.
                  sh 'fgpu/benchmarks/compute_bench.py --kernel all --passes 10'
                  sh 'fgpu/benchmarks/compute_bench.py --kernel ddc --mode=narrowband-discard --passes 10'
                  sh 'fgpu/benchmarks/compute_bench.py --kernel ddc --mode=narrowband-no-discard --passes 10'
                  sh 'fgpu/benchmarks/compute_bench.py --kernel pfb_fir --passes 10'
                  sh 'fgpu/benchmarks/compute_bench.py --kernel fft --passes 10'
                  sh 'fgpu/benchmarks/compute_bench.py --kernel postproc --passes 10'
                  sh 'fgpu/benchmarks/compute_bench.py --kernel all --mode=narrowband-discard --passes 10'
                  sh 'fgpu/benchmarks/compute_bench.py --kernel all --mode=narrowband-no-discard --passes 10'

                  sh 'fgpu/benchmarks/ddc_bench.py --passes 10'

                  sh 'fgpu/benchmarks/fft_bench.py --mode r2c --passes 10'
                  sh 'fgpu/benchmarks/fft_bench.py --mode c2c --passes 10'

                  sh 'xbgpu/benchmarks/beamform_bench.py --passes 10'
                  sh 'xbgpu/benchmarks/correlate_bench.py --passes 10'

                  sh './gpu_copy.py htod --repeat 10'
                  sh './gpu_copy.py dtoh --repeat 10'
                  sh './gpu_copy.py dtod --repeat 10'
                  // Fails on newer pycuda: https://github.com/inducer/pycuda/issues/459
                  // sh './gpu_copy.py htod --mem huge --fill 1 --repeat 10'
                }
              }
            }

            /* This stage ensures that all the python style guidelines checks pass.
             * This will catch if someone has committed to the repo without
             * installing the required pre-commit hooks, or has bypassed them.
             */
            stage('Run pre-commit checks') {
              steps {
                // no-commit-to-branch complains if we are on the main branch
                sh 'SKIP=no-commit-to-branch pre-commit run --all-files'
              }
            }

            /* This stage actually runs pytest. Pytest has a number of flags that are
             * not required but make life easier:
             * 1. -n X: Launches X processes and runs the tests in parallel across
             *     multiple processes. This speeds up testing significantly. NOTE: This
             *     can create resource contention over things like GPU RAM. If it
             *     starts becoming an issue set X to 1.
             * 2. -v: Increases verbosity
             * 3. --junitxml=reports/result.xml' Writes the results to a file for later
             *    examination.
             * 4. -m "not slow": skip slow tests
             */
            stage('Run pytest (quick)') {
              when { not { anyOf { changeRequest target: 'main'; branch 'main' } } }
              options { timeout(time: 30, unit: 'MINUTES') }
              steps {
                sh 'pytest -n 4 -v -ra -m "not slow" --junitxml=reports/result.xml --cov=katgpucbf --cov=test --cov-report=xml --cov-branch --suppress-tests-failed-exit-code'
              }
            }
            stage('Run pytest (full)') {
              when { anyOf { changeRequest target: 'main'; branch 'main' } }
              options { timeout(time: 60, unit: 'MINUTES') }
              steps {
                sh 'pytest -n 4 -v -ra --all-combinations --junitxml=reports/result.xml --cov=test --cov=katgpucbf --cov-report=xml --cov-branch --suppress-tests-failed-exit-code'
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
            recordCoverage sourceCodeEncoding: 'UTF-8', tools: [[parser: 'COBERTURA', pattern: 'coverage.xml']]
          }
        }
      }
    }

    // This stage runs directly on the host
    stage('Build and push Docker image') {
      when {
        not { changeRequest() }
        equals expected: "SUCCESS", actual: currentBuild.currentResult
      }
      steps {
        script {
          String branch = env.BRANCH_NAME
          String tag = (branch == "main") ? "latest" : branch
          // Supply credentials to Dockerhub so that we can reliably pull the base image
          docker.withRegistry("", "dockerhub") {
            dockerImage = docker.build(
              "harbor.sdp.kat.ac.za/dpp/katgpucbf:${tag}",
              "--pull "
              + "--label=org.opencontainers.image.revision=${env.GIT_COMMIT} "
              + "--label=org.opencontainers.image.source=${env.GIT_URL} ."
            )
          }
          docker.withRegistry("https://harbor.sdp.kat.ac.za/", "harbor-dpp") {
            dockerImage.push()
          }
          // Remove the built and pushed Docker image from host
          sh "docker rmi ${dockerImage.id}"
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
      body: '${SCRIPT, template="groovy-html.template"}',
      recipientProviders: [developers(), requestor(), culprits()],
      subject: '$PROJECT_NAME - $BUILD_STATUS!',
      to: '$DEFAULT_RECIPIENTS'

      cleanWs()
    }
  }
}
