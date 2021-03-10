job "katxgpu_job" {
  datacenters = ["brp"]

  group "fsim_host" {
    constraint {
      attribute = "${attr.unique.hostname}"
      value     = "qgpu01"
    }
    task "fsim" {
      driver = "docker"
      config {
        image = "katxgpu:fsim"
        network_mode = "host"
        privileged = true
        ulimit {
          memlock = -1
        }
        devices = [
          {
            host_path = "/dev/infiniband/rdma_cm"
          },
          {
            host_path = "/dev/infiniband/uverbs0"
          }
        ]
        cap_add = [
          "CAP_NET_RAW",
        ]
        command = "scratch/fsim"
        args = [
          "--interface", "10.100.43.1",
          "239.10.10.10:7149"
        ]
      }

      resources {
        memory = 32768 # Not really sure how much needs to be allocated, so I made it a lot.
      }
    }
  }

  group "katxgpu_host" {
    constraint {
      attribute = "${attr.unique.hostname}"
      value     = "qgpu02"
    }
    task "katxgpu_ingest" {
      driver = "docker"
      config {
        image = "katxgpu:fsim"
        network_mode = "host"
        privileged = true
        ulimit {
          memlock = -1
        }
        devices = [
          {
            host_path = "/dev/infiniband/rdma_cm"
          },
          {
            host_path = "/dev/infiniband/uverbs0"
          }
        ]
        cap_add = [
          "CAP_NET_RAW",
        ]
        command = "/usr/bin/python"
        args = [
          "scratch/receiver_example.py",
          "--mcast_src_ip", "239.10.10.10",
          "--mcast_src_port", "7149",
          "--src_interface", "10.100.44.1"
        ]
      }

      resources {
        memory = 32768
        device "nvidia/gpu" {
          count = 1
        }
      }
    }
  }
}
