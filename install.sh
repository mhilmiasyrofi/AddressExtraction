apt-get update
apt install tmux -y
pip install -r requirements.txt
python3 -c "import nltk; nltk.download('punkt')"