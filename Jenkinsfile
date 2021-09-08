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
    docker {
      /* This nvidia/cuda:11.4.1-devel-ubuntu20.04 docker image contains the CUDA 
       * development environment which is required to compile the kernels used
       * by PyCUDA. 
       */
      image 'nvidia/cuda:11.4.1-devel-ubuntu20.04'

      /* A number of arguments need to be  specified in order for the container
       * to launch correctly.
       *
       * --gpus=all: This argument passes the Nvidia driver and devices from the
       * host to the container. It requires the NVIDIA Container Runtime to be 
       * installed on the host.
       * 
       * --network=host: The docker container needs access to the internet to
       * install packages and an interface to transmit test networking code.
       * This command passes all the host interfaces to the container to be 
       * used as required.
       *
       * -u=root: The container needs to be root when running the "apt-get
       * install" commands. Additionally, programs running with ibverbs 
       * generally need to be run as root unless the CAP_NET_RAW capability
       * flag is set (Have not looked into how to do that). 
       *
       * -ulimit=memlock=-1: Not sure what this is for. The author of SPEAD2
       * recommended that this flag be here.
       *
       * --device=/dev/infiniband/rdma_cm and --device=/dev/infiniband/uverbs0:
       * These flags pass the drivers required for ibverbs to the container.
       * NOTE: The driver is not always ubverbs0, sometimes it is ubverbs1 or
       * 2 etc. The correct once needs to be specified.
       */
      args '--gpus=all --network=host -u=root --ulimit=memlock=-1 --device=/dev/infiniband/rdma_cm  --device=/dev/infiniband/uverbs0'
    }

  }
  
  environment {
    DEBIAN_FRONTEND = 'noninteractive' // Required for zero interaction when installing or upgrading software packages
  }

  /* This stage should ideally be part of the initial Docker image, as it
   * takes time and downloads multple Gigabytes from the internet. A new 
   * Dockerfile needs to be created that will extend the 
   * nvidia/cuda:11.4.1-devel-ubuntu20.04 image to include this install.
   */
  stages {
    stage('Configure Environment') {
      steps {
	sh 'apt-get update'
        sh 'apt-get install -y python3 python3-pip python3-dev python3-pybind11' // Required for python
        sh 'apt-get install -y git build-essential automake'
        sh 'apt-get install -y autoconf libboost-dev libboost-program-options-dev libboost-system-dev libibverbs-dev librdmacm-dev libpcap-dev' // Required for installing SPEAD2. Much of this is installed when using MLNX_OFED, TODO: Clarify
      }
    } 

    stage('Install required python packages') {
      steps {
        sh 'pip3 install -r requirements.txt -r requirements-dev.txt'
      }
    }
   
    /* This stage ensures that all the python style guidelines checks pass.
     * This will catch if someone has commited to the repo without installing 
     * the required pre-commit hooks.
     */
    stage('Check pre-commit hooks have been applied') {
      steps {
        sh 'pre-commit install'
        sh 'pre-commit run --all-files'
      }
    }

    stage('Install katgpucbf package') {
      steps {
        sh 'pip3 install .'
      }
    }

    /* This test verifies that the fsim can send a burst of traffic on the
     * network. Currently no unit tests run on the network, so this stage
     * serves to catch a few issues that could be missed by the unit tests when
     * it comes to networking  This stage verifies two things:
     * 1. SPEAD2 normally installs with ibverbs settings enabled but under some
     *    conditions SPEAD2 will not install ibverbs functions. When running
     *    make on the fsim, an error will be thrown if SPEAD2 does not install
     *    correctly.
     * 2. The commands required to to run a docker container that makes use of
     *    ibverbs are not trivial to determine. Attempting to send a burst of
     *    data out on the network with the fsim will quickly reveal if there are
     *    any issues with mechanism.
     */
    stage('Test fsim compilation') {
      steps {
        // Install SPEAD2 C++ library required for installation of fsim
        dir('3rdparty/spead2') {
          sh './bootstrap.sh'
          sh './configure'
          sh 'make'
          sh 'make install'
        }
        // Make and compile fsim.
        dir('src/tools') {
          sh 'make clean'
          sh 'make -j dsim fsim'  
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
    stage('Run pytest fgpu & xgpu') {
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
          to: 'ijassiem@ska.ac.za'
		    
	  cleanWs()
        }
      }
  
}
