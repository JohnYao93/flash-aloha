conda env create -f src/hardware/arx5-sdk/conda_environments/py310_environment.yaml
conda activate arx-py310
mkdir -p src/hardware/arx5-sdk/build && cd src/hardware/arx5-sdk/build
cmake ..
make -j

cd ../../../..



