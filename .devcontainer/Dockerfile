# Use the official Python 3.14 slim image for a smaller, production-ready footprint
FROM python:3.14-slim

# Python best practices in Docker
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Create a non-root user for development
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo git wget curl abcmidi \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && rm -rf /var/lib/apt/lists/*

# Install locales and generate en_US.UTF-8
RUN apt-get update && apt-get install -y locales \
    && sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen \
    && locale-gen

# Set environment variables for the locale
ENV LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    LC_ALL=en_US.UTF-8

# Upgrade pip and install pip-tools globally
RUN python -m pip install --upgrade pip pip-tools

# Set the working directory and switch to the non-root user
WORKDIR /workspace
USER $USERNAME

# Add the local user's bin folder to the PATH
ENV PATH="/home/$USERNAME/.local/bin:${PATH}"

# Copy the pre-compiled lockfile from the root context
COPY --chown=$USERNAME:$USER_GID requirements.txt /tmp/

# Install exactly what is locked
RUN pip install --user -r /tmp/requirements.txt

# Add local user bin to PATH so installed packages are executable
ENV PATH="/home/$USERNAME/.local/bin:${PATH}"
