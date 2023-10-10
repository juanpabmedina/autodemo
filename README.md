
# VisualDemo-Cho

VisualDemo-Cho is an automatic design method that combines apprenticeship learning with AutoMoDe-Chocolate, a state-of-the-art automatic off-line design method.
VisualDemo-Cho generate control software for robot swarms based on images demonstrations of the desire behaviour.

## Installation

### Dependencies:

* [ARGoS3-AutoMoDe](https://github.com/demiurge-project/ARGoS3-AutoMoDe)
* If you are on the cluster [install ARGoS3-AutoMoDe like this](https://iridia.ulb.ac.be/wiki/Getting_started_for_Demiurge)
* A python enviroment with requirements.txt install: 
    ```PowerShell
    pip install -r requirements.txt
    ```

### Orchestra:

Once you installed `ARGoS3-AutoMoDe`, open the `experiment-loop-function` folder and paste `orchestra` folder. Then add `add_subdirectory(orchestra)` in the CMakeLists.txt and recompile with Cmake instructions for `experiment-loop-function`.

## How to use 

### Files locations
* Keep the content of `local_code` on your local compute.
* Load the another content on your cluster.

### Running an experiment for Demo-Cho
1. Run `python3 runAutodemo.py [missionName_date] [number of independent designs] [number of iterations]` to launch the design process on the cluster.
2. Whait for the cluster to finished to run the design process.
3. Run `python3 JsonCollector.py [mission name + date] [number of independent designs]` to collect all data of the design process in for of Json files.