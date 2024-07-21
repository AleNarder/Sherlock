FROM osrf/ros:humble-desktop-full-jammy as base

RUN apt-get update && apt-get install -y python3-pip

# Add vscode user with same UID and GID as your host system
# (copied from https://code.visualstudio.com/remote/advancedcontainers/add-nonroot-user#_creating-a-nonroot-user)
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN sudo apt update && sudo apt install locales
RUN sudo locale-gen en_US en_US.UTF-8
RUN sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
RUN export LANG=en_US.UTF-8

# Prevent GPG key errors
# @see https://github.com/osrf/docker_images/issues/697
# @see https://discourse.ros.org/t/again-snapshot-repo-gpg-key-expired/34733/6
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 4B63CF8FDE49746E98FA01DDAD19BAB3CBF125EA

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Switch from root to user
USER $USERNAME

# Add user to video group to allow access to webcam
RUN sudo usermod --append --groups video $USERNAME

# Rosdep update
RUN rosdep update

# Source the ROS setup file
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc


# Install X11 for GUI
RUN sudo apt-get install -y \
    xauth \
    x11-apps \
    libxext-dev \
    libxrender-dev \
    libxtst-dev

RUN git config --global --add safe.directory "*"

# Install taskfile
RUN sudo sh -c "$(curl --location https://taskfile.dev/install.sh)" -- -d

#################################
# DEVELOPMENT TARGET
#################################
FROM base as dev

ENV BAGS_URL=${CALIBRATION_BAGS_URL}

# Install gazebo
RUN sudo apt-get update && sudo apt-get install -y lsb-release wget gnupg curl
RUN sudo wget https://packages.osrfoundation.org/gazebo.gpg -O /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null \
    && sudo apt-get update \
    && sudo apt-get install -y \gz-harmonic 

# Install Node.js
RUN sudo curl -fsSL https://deb.nodesource.com/setup_22.x -o nodesource_setup.sh   
RUN sudo -E bash nodesource_setup.sh && sudo apt-get install -y nodejs

# Update PATH
RUN echo "source /calibration/ros2/install/setup.bash" >> ~/.bashrc

# Install git-lfs
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash && sudo apt-get install git-lfs

#################################
# PRODUCTION TARGET
#################################
FROM base as prod