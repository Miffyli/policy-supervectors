Originally cloned from https://github.com/uber-research/deep-neuroevolution and modified to fit needs of our project.

## Running the novelty search experiment
  To create env (note that the name is important):
  ```
  conda create -n neuroevol python=3.6
  conda activate neuroevol
  pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
  ```

  Add execution priviledges `chmod +x run_pointenv.sh scripts/*`.

  Then run `run_pointenv.sh` scripts in the root of this directory (with `neuroevol` environment activated).


  **Important notes**
  - You need to install `redis-server` for the code to work (see `README_original.md`)
  - You may need to stop an existing redis-server (e.g. on Ubuntu `/etc/init.d/redis-server stop`)
  - `run_pointenv.sh` continuously kills **all** redis-servers with ``killall redis-server``.
  - **`run_pointenv.sh` only works in the context of this project, i.e. it expects files/directories to exist above this directory**
  - Results files are put into `../experiments/novelty_*`

  **See README_original.md for the original version of README**
