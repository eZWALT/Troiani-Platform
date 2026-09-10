from troiani_platform.discovery.gpu import discover_gpus, infer_model

XML = """<?xml version="1.0"?>
<nvidia_smi_log>
  <gpu>
    <product_name>NVIDIA A100-PCIE-40GB</product_name>
    <uuid>GPU-aaa</uuid>
    <fb_memory_usage><total>40960 MiB</total><used>26000 MiB</used></fb_memory_usage>
    <utilization><gpu_util>12 %</gpu_util><memory_util>40 %</memory_util></utilization>
    <temperature><gpu_temp>30 C</gpu_temp></temperature>
    <processes>
      <process_info><pid>1</pid><process_name>python</process_name><used_memory>25000 MiB</used_memory></process_info>
    </processes>
  </gpu>
  <gpu>
    <product_name>NVIDIA A100 80GB PCIe</product_name>
    <uuid>GPU-bbb</uuid>
    <fb_memory_usage><total>81920 MiB</total><used>1 MiB</used></fb_memory_usage>
    <utilization><gpu_util>0 %</gpu_util></utilization>
    <temperature><gpu_temp>25 C</gpu_temp></temperature>
    <processes></processes>
  </gpu>
</nvidia_smi_log>
"""


def test_infer_model():
    assert infer_model("NVIDIA A100-PCIE-40GB") == "A100"
    assert infer_model("NVIDIA A100 80GB PCIe") == "A100"


def test_parse_xml_heterogeneous():
    gpus = discover_gpus(node="atlas", xml=XML, owned_users=["wtroi"])
    assert len(gpus) == 2
    assert gpus[0].memory_gb == 40
    assert gpus[1].memory_gb == 80
    assert gpus[0].uuid == "GPU-aaa"
    assert gpus[0].researcher_present
    assert gpus[1].safe_to_allocate()
    assert gpus[0].sm_util == 12
    assert gpus[0].memory_util == 40
    assert round(gpus[0].vram_pct) == 63
    payload = gpus[0].to_dict()
    assert payload["sm"] == 12
    assert payload["vram_used_gb"] == gpus[0].memory_used_gb
    assert payload["ref"] == "atlas/gpu0"
