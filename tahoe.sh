python3 -m venv tahoe
source tahoe/bin/activate
pip install jupyter
pip install ipykernel
python -m ipykernel install --user --name=tahoe --display-name "tahoe"
pip install pandas
pip install scanpy
pip install pyarrow
pip install gcsfs
pip install google-cloud-storage
pip install fsspec