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
      /* Use the initial layers of the build image as the image for testing. This
       * provides the appropriate dependencies.
       */
      additionalBuildArgs '--target=build-base'

      /* The following argument needs to be specified in order for the container
       * to launch correctly.
       *
       * --gpus=all: This argument passes the Nvidia driver and devices from the
       * host to the container. It requires the NVIDIA Container Runtime to be
       * installed on the host.
       */
      args '--gpus=all'
    }

  }

  environment {
    DEBIAN_FRONTEND = 'noninteractive' // Required for zero interaction when installing or upgrading software packages
    GIT_AUTHOR_EMAIL = sh(script: "git --no-pager show -s --format=\'%ae\'", returnStdout: true)
  }

  stages {
    stage('Install Python packages') {
      steps {
        sh 'pip install -r requirements.txt -r requirements-dev.txt'
      }
    }

    stage('Parallel stage') {
      parallel {
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

        stage('Install katgpucbf package') {
          steps {
            sh 'pip install --no-deps . && pip check'
          }
        }

        /* This stage verifies the successful installation of SPEAD2.
         *
         * SPEAD2 normally installs with ibverbs settings enabled but under some
         * conditions SPEAD2 will not install ibverbs functions. When running
         * make, an error will be thrown if SPEAD2 does not install
         * correctly.
         */
        stage('Compile C++ tools') {
          steps {
            // Install SPEAD2 C++ library required for installation of tools
            dir('3rdparty/spead2') {
              sh './bootstrap.sh'
              sh './configure'
              sh 'make -j'
              sh 'make install'
            }
            // Make and compile tools.
            dir('src/tools') {
              sh 'make clean'
              sh 'make -j'
            }
          }
        }
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
      steps {
        sh 'pytest -v -rs --junitxml=reports/result.xml'
      }
    }
    stage('Run pytest (full)') {
      when { anyOf { changeRequest target: 'main'; branch 'main' } }
      steps {
        sh 'pytest -v -rs --all-combinations --junitxml=reports/result.xml'
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
      subject: '$PROJECT_NAME - $BUILD_STATUS!',
      to: '$DEFAULT_RECIPIENTS, cc:' + GIT_AUTHOR_EMAIL

      cleanWs()
    }
  }

}
