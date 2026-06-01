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
