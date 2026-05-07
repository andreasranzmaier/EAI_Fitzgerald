# Dev Container Setup

## Prerequesites
- Docker is running on your system
- Dev Container Extension is installed (You can use the profile from CPP3)

## Configuration
- Enter the IP or your Pi and your username (kit-XX) in the .env file
- Build and open the container (ensure the Pi is connected when the container is starting)
- During the first build the container will prompt you for the password of your Pi (by default following the pattern kit-XX), please enter the password
- If you mistyped your password or the initialization sequence exited with an error for any other reason, please reinitialize by executing the following command inside the devcontainer
```
bash .devcontainer/init.sh
```

## Deployment and Debugging
All targets for building and deployment can be run via tasks in VSCode.
- ```train-model``` executes the Python script to train a TensorFlow model and convert it to TensorFlow Lite
- ```build``` builds the C++ application for model inference
- ```deploy``` connects to the Pi and starts debugging (and if necessary rebuilds the C++ application and trains the model)
- ```console``` opens a terminal that forwards the output from the C++ application running on the Pi to your devcontainer