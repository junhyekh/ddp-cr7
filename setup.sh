#!/bin/bash
set -e

# Step 1: Get input username, create user with a home directory, and add to group 'mamba'
read -p "Enter new username: " username

# Create the user with a home directory and add to group 'mamba'
sudo useradd -m -s /bin/bash "$username" && echo "User '$username' created and added to group 'mamba'."

# Step 2: Copy the /root/ddp folder to the new user's home directory
user_home="/home/$username"
if [ -d /root/ddp-cr7 ]; then
    sudo cp -r /root/ddp-cr7 "$user_home/ddp-cr7"
    # Change ownership so the new user can manage the folder
    sudo chown -R "$username":"$username" "$user_home/ddp-cr7"
    echo "Folder '/root/ddp-cr7' copied to $user_home."
else
    echo "Error: /root/ddp-cr7 does not exist."
    exit 1
fi

# Step 3: Create the micromamba environment using the new user's name
# Run the micromamba command as the new user so that the environment is created in their context.
# source /usr/local/bin/_activate_current_env.sh
# eval "$("${MAMBA_EXE}" shell hook --shell=bash)"
# micromamba activate "${ENV_NAME:-base}"
env_path="$MAMBA_ROOT_PREFIX/envs/ddp-$username"
sudo micromamba create --prefix "$env_path" -f /opt/workspace/ddp_base.yaml --yes
echo "Micromamba environment ddp-$username created."

# Step 4: Switch to the new user and change the current directory to the user's home directory
# echo "Switching to user '$username'..."
# # exec su "$username" -c "micromamba activate ddp-$username && cd \$HOME"
# # exec cd "$HOME"
# # sudo "micromamba deactivate"
# cd /home/$username/ddp-cr7
# sudo "micromamba activate /opt/conda/envs/ddpddp-$username"
# -c "exec bash && micromamba activate ddp-$username && cd \$HOME"
# #  && exec bash"
# exec "pip install -e /home/$username/ddp-cr7"
