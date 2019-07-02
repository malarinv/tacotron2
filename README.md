
## Setup
- clone the repo

`git clone https://github.com/agaralabs/tacotron2`
- cd to `tacotron2` copy models from wolverine:

`scp wolverine:/home/ubuntu/tacotron2/{checkpoint_15000,waveglow_256channels.pt} ./`

`scp wolverine:/home/ubuntu/tacotron2/waveglow ./`

**Wolverine Details:**
```
Host wolverine
    Hostname 54.71.137.17
    User ubuntu
    IdentityFile ~/.ssh/id_hip_ml
```
install the dependencies
`pip install requirements.txt`

## Running:
`python final.py`
