#!/bin/bash
set -e


# Switch to the new user and change the current directory to the user's home directory
echo "Switching to user '${USER}'..."
cd ~

# Activate the micromamba environment named "{username}-ddp"
# Here, ${USER} is used to automatically get the current username.
eval "$(micromamba shell hook --shell=bash)"

micromamba activate "ddp-${USER}"

# Install the ddp-cr7 package in editable mode from the home directory
cd ~/ddp-cr7
git init
pip install -e ~/ddp-cr7
# # exec su "$username" -c "micromamba activate ddp-$username && cd \$HOME"
# # exec cd "$HOME"
# # sudo "micromamba deactivate"
# cd /home/$username/ddp-cr7
# sudo "micromamba activate /opt/conda/envs/ddpddp-$username"
# -c "exec bash && micromamba activate ddp-$username && cd \$HOME"
# #  && exec bash"
# exec "pip install -e /home/$username/ddp-cr7"
