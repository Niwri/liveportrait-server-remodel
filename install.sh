conda create -n LivePortrait python=3.10
conda activate LivePortrait
pip install -r requirements.txt
hf download KlingTeam/LivePortrait --local-dir pretrained_weights --exclude "*.git*" "README.md" "docs"
