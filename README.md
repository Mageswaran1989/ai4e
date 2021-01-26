# ai4e
Artificial Intelligence For Engineers


## Python environment

```
sudo apt install libblas3 liblapack3 liblapack-dev libblas-dev
sudo apt-get install libcairo2-dev libjpeg-dev libgif-dev # #B1B maninlib cairo dependency
sudo apt install ffmpeg
#https://github.com/ppaquette/gym-doom
sudo apt-get install -y python-numpy cmake zlib1g-dev libjpeg-dev libboost-all-dev gcc libsdl2-dev wget unzip git libbz2-dev
sudo apt-get install -y python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb ffmpeg xorg-dev python-opengl libboost-all-dev libsdl2-dev swig
sudo apt-get install -y python-numpy cmake zlib1g-dev libjpeg-dev libboost-all-dev gcc libsdl2-dev wget unzip

conda create --prefix /opt/envs/ai4e/ python==3.8
pip install -r requirements.txt
conda install -c plotly plotly-orca # Plotly static images

```


**Jupyter**

- https://github.com/plotly/jupyter-dash
- https://stackoverflow.com/questions/52771328/plotly-chart-not-showing-in-jupyter-notebook
- https://stackoverflow.com/questions/56843745/automatic-cell-execution-timing-in-jupyter-lab
```
pip install jupyter-dash
jupyter labextension install jupyterlab-plotly
jupyter labextension install @jupyter-widgets/jupyterlab-manager plotlywidget
jupyter labextension install jupyterlab-execute-time
```


**Rapids**

```
#https://github.com/rapidsai/cudf
conda install -c rapidsai -c nvidia -c numba -c conda-forge \
    cuml=0.16 python=3.7 cudatoolkit=10.2

conda install -c rapidsai -c nvidia -c numba -c conda-forge \
    rapids=0.16 python=3.7 cudatoolkit=10.2

#https://rapids.ai/start.html#rapids-release-selector
conda create --prefix /opt/envs/ai4e -c rapidsai -c nvidia -c conda-forge \
    -c defaults rapids=0.16 python=3.8 cudatoolkit=10.2

conda install -c rapidsai -c nvidia -c conda-forge \
    -c defaults rapids=0.16 python=3.8 cudatoolkit=10.2



```

## RL

```

```
