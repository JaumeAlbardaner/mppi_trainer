# mppi_trainer
This repository contains tools to train a neural network that describes the behavior of the dynamic model of the NinjaCar. It can be used to train any neural network that is to be used by the [MPPI algorithm](https://github.com/JaumeAlbardaner/ninjacar_mppi).

## Contents:

1. [Resulting NN](#1-resulting-nn)
2. [Usage](#2-usage)


## 1. Resulting NN:
### 1.1 Input (vehicle frame):

·   **roll**: Angular velocity of the vehicle along the longitudinal axis of the vehicle (the direction it's running towards).

·   **u_x**: Longitudinal velocity of the vehicle.

·   **u_y**: Lateral velocity of the vehicle.

·   **yaw_der**: Angular acceleration of the vehicle along its vertical axis (the axis perpendicular to the ground).

·   **steering**: Steering input given to the vehicle.

·   **throttle**: Throttle input given to the vehicle.

### 1.2 Output:

·   **u_x_der**: Increment of position along the *global* x axis.

·   **u_y_der**: Increment of position along the *global* y axis.

·   **yaw_der_der**: Negated *yaw_der*.

**WARNING:** The *yaw_der* is negated because their mocap space provides the negative value of the yaw derivative (as they explain in [this line](https://github.com/AutoRally/autorally/blob/48bae14fe4b2d56e5ca11fd2fec3d7dfac7a0e75/autorally_control/include/autorally_control/path_integral/neural_net_model.cu#L196)), it may not be necessary in our case. Nevertheless, left as-is has worked with the [pre-trained model](https://github.com/AutoRally/autorally/blob/melodic-devel/autorally_control/src/path_integral/params/models/autorally_nnet_09_12_2018.npz).

·   **roll_der**: Roll derivative.

### 1.3 Source:

As the official AutoRally repository does not properly document how the neural network has to be trained, these inputs and outputs were deduced.

[This line](https://github.com/AutoRally/autorally/blob/c2692f2970da6874ad9ddfeea3908adaf05b4b09/autorally_control/include/autorally_control/path_integral/neural_net_model.cu#L192-L197) describes how some of the outputs are calculated from the state variable, and the state variable is described in [this line](https://github.com/AutoRally/autorally/blob/c2692f2970da6874ad9ddfeea3908adaf05b4b09/autorally_control/include/autorally_control/path_integral/run_control_loop.cuh#L119).

The outputs were verified to be as described by outputting the actual values that were being processed by the algorithm. Script *visualizer.py* (from commit [4eb52ea451](https://github.com/JaumeAlbardaner/mppi_trainer/tree/4eb52ea4513f022f16eaafde5b56f10ec7a15b3a)), and performing a visual inspection on them.

These also make sense because of how they are used to update the state of the vehicle in [updateState](https://github.com/AutoRally/autorally/blob/48bae14fe4b2d56e5ca11fd2fec3d7dfac7a0e75/autorally_control/include/autorally_control/path_integral/neural_net_model.cu#L282), a sum: current state + derivative * time increment.

## 2. Usage:

This section is reserved to explain how to extract the information from the rosbag files and turn it into a working neural network.

### 2.1 Rosbag creation:

1. As explained in point 5 of the _sidenotes_ section of the [ninjacar_mppi](https://github.com/JaumeAlbardaner/ninjacar_mppi) repository, the [command_rosbag.py](https://github.com/JaumeAlbardaner/ninjacar_mppi/blob/master/autorally_control/src/path_integral/scripts/command_rosbag.py) script must be executed.

2. Make sure the topics `/vehicle_controls` and `/odom_reframer/odom_chassis` (or whatever you decided to name them) are publishing something. Afterwards, run:
    ```
    rosbag record /vehicle_controls /odom_reframer/odom_chassis
    ```
    Remember to source into the workspace, otherwise `/vehicle_controls` may not be visible, as it is a custom ROS message.

3. Drive the car around, and record a **train** and **test** bag!

### 2.2 Extraction of the data from rosbag

For this purpose an [external repository](https://github.com/rdesc/autorally/tree/rdesc-melodic-devel/autorally_control/src/path_integral/scripts/ml_pipeline) was used. It was initially used for the entire _bag extraction_ + _NN training_, because of its theoretically [good results](https://rdesc.dev/vcr_final_presentation.pdf). However, the input and output that it was using to train the NN was not the one that was explained in [Section 1](#1-resulting-nn). Thereafter, it was kept just for its rosbag extracting capabilities.

1.  Clone this repo:
    ```
    cd ~/Documents/ && git clone git@github.com:JaumeAlbardaner/mppi_trainer.git
    ``` 

2.  Make sure you have all the required python3 libraries:
    ```
    cd mppi_trainer
    pip3 install -r requirements.txt
    ```

3.  Extract data from rosbags:
    Modify the first three lines from [config.yml](https://github.com/JaumeAlbardaner/mppi_trainer/blob/master/ml_pipeline/config.yml)

    ·   **rosbag_filepath**: location of the rosbag file to be extracted (e.g: ./test.bag).

    ·   **final_file_name**: name of the final csv file where the data will be extracted (e.g: test.csv).

    ·   **run_name**: folder within _pipeline_files_ where plots and the csv files will be stored (e.g: N03_test).

    This must be performed twice (once for the train bag and again for the test bag):
    
    ```
    cd ml_pipeline
    python3 trainer.py
    ```

### 2.3 NN Training:

1.  Preprocess the data:
    Establish the limits to your data in [preprocess.py](https://github.com/JaumeAlbardaner/mppi_trainer/blob/b9f4ccc63cfa0aa032b7a3ea56c69edf545c2532/preprocess.py#L27-L53), and perform this command for both the train and test csv files:
    ```
    cd ..
    python3 preprocess.py ml_pipeline/pipeline_files/N03_train/data/train.csv
    ```

2.  Train your neural network:
    ```
    python3 train.py train_preprocessed.csv test_preprocessed.csv
    ``` 

    **UNTESTED RESULTS** For enhanced results, insted of starting from a weird initialization, you may start training your neural network from the AutoRally's one by calling:
    ```
    python3 train.py train_preprocessed.csv test_preprocessed.csv autorally.npz
    ``` 

This procedure should yield a custom.npz file, as well as a `Errors.png` graph displaying the evolution of the errors.

To change the learning rate, change the value on line [#36](https://github.com/JaumeAlbardaner/mppi_trainer/blob/b9f4ccc63cfa0aa032b7a3ea56c69edf545c2532/train.py#L36), and to change the number of epochs, change the value on line [#116](https://github.com/JaumeAlbardaner/mppi_trainer/blob/b9f4ccc63cfa0aa032b7a3ea56c69edf545c2532/train.py#L116).

### 2.4 NN Usage:

1.  Copy the NN to the MPPI directory where it's supposed to be loaded from:
    ```
    cp custom.npz ~/catkin_ws/src/ninjacar_mppi/autorally_control/src/path_integral/params/models/
    ``` 

2.  Modify the desired `.launch` file, setting the **model_path** variable to the following value:
    ```
    "$(env AR_MPPI_PARAMS_PATH)/models/custom.npz"
    ```

3.  Roslaunch it!

## Sidenotes:

1.  If the `train.py` script returns:
    ```
    ValueError: Can't convert Python sequence with mixed types to Tensor.
    ```
    comment lines 61 and 67 (kernel_initializers)

2. To observe in real time how the mean square error is evolving during training, execute the following command:
    ```
    tensorboard --logdir logs/
    ```