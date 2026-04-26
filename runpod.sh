python3 -m venv venv_ssl_proj
source venv_ssl_proj/bin/activate
pip cache purge
pip install --no-cache-dir torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu126
pip3 install kagglehub matplotlib dotenv timm wildlife-datasets kornia
pip3 install wildlife-datasets git+https://github.com/WildlifeDatasets/wildlife-tools --quiet --upgrade-strategy only-if-needed
git config user.email alfred.andrei@yahoo.com
git config user.name AlfredPietraru
python3 main_utils.py