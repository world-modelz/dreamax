apt-get update && apt-get install git curl wget vim less unzip unrar htop iftop iotop build-essential autotools-dev nfs-common pdsh cmake g++ gcc ca-certificates ssh python3-dev libpython3-dev python3-pip -y
apt-get install ffmpeg xvfb libsm6 libxext6 libx11-6 libgl1-mesa-glx libosmesa6 mesa-utils swig python3-opengl -y
update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1 && pip install --upgrade pip

sudo pip freeze | grep -v "^-e" | grep -v -E "blinker|colorama|command-not-found|entrypoin|Cython|iotop|pexpect|simplejson|pyasn1|PyYAML|sos|systemd-p|ufw" | xargs sudo pip uninstall -y

python3 -m pip --no-cache-dir install atari_py autorom[accept-rom-license] gym[atari] gym[box2d] gym[classic_control] gym[toy_text]
python3 -m pip --no-cache-dir install autorom && AutoROM --accept-license

cat ./requirements.txt | grep -E -v "tensorflow|dm-control|tf-agents==|jax" > /tmp/requirements.txt

python3 -m pip --no-cache-dir install -r /tmp/requirements.txt
python3 -m pip --no-cache-dir install numpy scipy six wheel jax[tpu]==$( cat ./requirements.txt | grep "jax==" | grep -oE "(\w*[.]\w*)*") -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
python3 -m pip --no-cache-dir install $( cat ./requirements.txt |grep "tensorflow" | grep -v "datasets" | grep -o '^[^#]*')

cat ./requirements.txt | grep -E "tensorflow|dm-control|tf-agents" | grep -v "pygame" > /tmp/requirements.txt

python3 -m pip --no-cache-dir install -r /tmp/requirements.txt

rm /tmp/requirements.txt