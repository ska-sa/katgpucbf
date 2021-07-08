# This file describes a nomad "job", which is similar to what Docker would call a "stack" or kubernetes a "deployment".
# In order to use it, the following command should be run on the nomad server:
# `nomad job plan katxgpu.nomad`
# This will tell you whether or not we are good to go ahead, and if so, it'll give you the command to run in order
# to put the planned job into action.

# This particular job launches an fsim on qgpu01, and a katxgpu spead receiver on qgpu02.

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
        # Here I tag the image with a specific "version" to prevent nomad from trying to pull the image from a non-
        # existant repo. This way, it uses just the one that is already on the host, with this tag.
        image = "katxgpu:fsim"
        # Tell docker to use the host network. There may be a way to not have to do this, but currently we don't know
        # it. Perhaps it is the most elegant solution for telling containers which interface to use for the multicast
        # data streams.
        network_mode = "host"
        # docker doesn't provide capabilities to non-root users. Haven't found a better workaround yet than just giving
        # root access to the container. For us there aren't really security implications though so it should be fine.
        privileged = true
        ulimit {
          memlock = -1
        }
        devices = [
          {
            host_path = "/dev/infiniband/rdma_cm"
          },
          {
            # It's worth noting that this device can sometimes be called uverbs1 or uverbs2. Just check.
            host_path = "/dev/infiniband/uverbs0"
          }
        ]
        # spead2 uses ibverbs, and ibverbs needs this capability.
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
        # For the time being, we don't bother assigning CPU resources. Uses nomad defaults.
        # At this point it is not clear to me whether nomad actually enforces the limits, or whether they are just
        # used for scheduling and placement.
        memory = 4096 # Unit is MB.
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
        memory = 4096
        # This allocates a GPU to the container that the task runs in.
        device "nvidia/gpu" {
          count = 1
        }
      }
    }
  }
}
