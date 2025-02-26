#!/bin/bash
set -e

# Step 1: Get input username, create user with a home directory, and add to group 'mamba'
read -p "Enter new username: " username

# Create the user with a home directory and add to group 'mamba'
sudo useradd -m -s /bin/bash -G mamba "$username" && echo "User '$username' created and added to group 'mamba'."

# Step 2: Copy the /root/ddp folder to the new user's home directory
user_home="/home/$username"
if [ -d /root/ddp-cr7 ]; then
    sudo cp -r /root/ddp-cr7 "$user_home/"
    # Change ownership so the new user can manage the folder
    sudo chown -R "$username":"$username" "$user_home/ddp-cr7"
    echo "Folder '/root/ddp-cr7' copied to $user_home."
else
    echo "Error: /root/ddp-cr7 does not exist."
    exit 1
fi

# Step 3: Create the micromamba environment using the new user's name
# Run the micromamba command as the new user so that the environment is created in their context.
sudo micromamba create -n ddp-"$username" -f /opt/workspace/ddp_base.yaml --yes
echo "Micromamba environment ddp-$username created."

# Step 4: Switch to the new user and change the current directory to the user's home directory
echo "Switching to user '$username'..."
exec su - "$username"
#  -c " export ENV_NAME=ddp-$username \
# && micromamba activate ddp-$username && cd \$HOME && exec bash"
# exec "pip install -e \$HOME/ddp-cr7"
