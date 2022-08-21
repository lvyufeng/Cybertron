import os

os.system("git clone https://github.com/lvyufeng/Cybertron")
os.chdir("Cybertron")
os.system("conda create -n mindspore python=3.7.5 cudatoolkit=11.1 cudnn -y")
os.system("/opt/conda/envs/mindspore/bin/pip install -r requirements/gpu_requirements.txt")
return_code = os.system("/opt/conda/envs/mindspore/bin/pytest tests")
if return_code:
    raise Exception("tests failed.")