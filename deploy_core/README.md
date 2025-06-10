# Deploy Core

The `DeployCore` module defines the abstract functionalities for all components, including core inference capabilities, 2D detection features, SAM functionalities, and more. Beyond merely defining abstract functions, DeployCore also provides external encapsulations for certain algorithms. When implementing an algorithm, developers only need to focus on completing the key processes outlined in these definitions to achieve seamless algorithm deployment.

## Functionality

`DeployCore` is designed to provide abstract interface definitions for the functionalities of all modules, as well as abstract base classes containing reusable code.

- Abstract core inference functionality: `BaseInferCore`
- Abstract 2D detection functionality: `BaseDetection2DModel`
- Abstract SAM functionality: `BaseSamModel`
- Plug-and-play asynchronous pipeline base class: `BaseAsyncPipeline`

## Structure

The entire project code is divided into three parts:
  1. Abstract interface classes for functional modules
  2. Abstract base classes for certain functional modules
  3. Base classes for the asynchronous inference pipeline framework

code structure:
  ```bash
  deploy_core
  |-- CMakeLists.txt
  |-- README.md
  |-- include
  |   `-- deploy_core
  |       |-- base_infer_core.hpp
  |       |-- base_detection.hpp
  |       |-- base_sam.hpp
  |       |-- async_pipeline.hpp
  |       |-- async_pipeline_impl.hpp
  |       |-- block_queue.hpp
  |       |-- common.hpp
  |       `-- wrapper.hpp
  `-- src
      |-- base_detection.cpp
      |-- base_infer_core.cpp
      `-- base_sam.cpp
  ```


  - Abstract interface classes for functional modules
    ```bash
    |-- base_infer_core.hpp
    |-- base_detection.hpp
    |-- base_sam.hpp
    ```
    1. **`base_infer_core.hpp`**: Defines the core inference functionalities and related abstract classes, while also providing an abstract base class for the foundational features of the inference core module.
    2. **`base_detection.hpp`**: Defines the abstract base class for 2D detection functionalities.
    3. **`base_sam.hpp`**: Defines the abstract base class for SAM functionalities.

  - Base classes for the asynchronous inference pipeline framework
    ```bash
    |-- async_pipeline.hpp
    |-- async_pipeline_impl.hpp
    |-- block_queue.hpp
    |-- common.hpp
    `-- wrapper.hpp
    ```
    1. **`async_pipeline.hpp`** and **`async_pipeline_impl.hpp`**: Define the asynchronous inference framework and its implementation.
    2. **`block_queue.hpp`**: Implements the blocking queue.
    3. **`common.hpp`**: Contains common definitions, such as 2D bounding boxes.
    4. **`wrapper.hpp`**: Provides wrappers for certain classes, such as the encapsulation of OpenCV's `cv::Mat` format.


## TODO
