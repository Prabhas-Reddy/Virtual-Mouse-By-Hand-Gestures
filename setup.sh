#!/bin/bash

# Uninstall existing OpenCV version
pip uninstall -y opencv-python-headless

# Install the specific version of OpenCV
pip install opencv-python-headless==4.8.0.76

# Install other dependencies listed in requirements.txt
pip install -r requirements.txt
