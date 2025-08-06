# faab-hyperparams

this code has been tested with Bela at `dev` commit `71a6e62a`

Instructions to set up the Bela are in [setup-bela.md](setup-bela.md). If you have been given a microsd card by me, you can skip them. Just open the Bela IDE (type `bela.local`in the browser) and run the `faab-run` project.

# Setup

To use `pyaudio` we need `portaudio` installed. In mac (for other platforms see [portaudio installation instructions](https://pypi.org/project/PyAudio/))

```bash
brew install portaudio
```

This project uses `uv` to manage dependencies

```bash
pip install uv
```

Clone this repository with submodules:

```bash
git clone --recurse-submodules -j8  git@github.com:pelinski/ai-mami-faab.git
```

Install the dependencies with `uv`:

```bash
cd faab-hyperparams # or the folder where you cloned the repo
uv venv
uv pip install torch #==2.0.1+cu117 --index-url https://download.pytorch.org/whl/cu117 # for g15
```

## Download trained models

Download the trained models from [this link](https://www.dropbox.com/scl/fo/3ou91kqehjeb9g30roslq/AM_sA7LBkurbI3OJTwrcx2Y?rlkey=8q9chh9cgiy5nqye6hzh8xdtp&st=onfmepsq&dl=0) and extract them in the `src/models` folder.

# Running the python code

If it's not running already, run the `faab-run` project from the Bela IDE.

**Important**: If you stop the python code, you should stop the Bela project as well, otherwise it will give a `Buffer n full` warning. If you want to restart the python code, you should restart the Bela project first.

## for pepper

Inside the repo folder, run the following command:

```bash
# in the ai-mami-faab folder
pipenv run callback
```

## with OSC + SuperCollider

The SuperCollider code to receive the OSC messages is in the `supercollider` folder. Run it in SuperCollider.
To run the python code, inside the repo folder, run:

```bash
# in the ai-mami-faab folder
pipenv run callback-osc
```

# Troubleshooting

for the Pepper mode, if you see "Buffer n full" in the Bela console, it means that the data is not being sent quickly enough to Bela. This can happen because you stopped the python code and the Bela code is still running. If this is not the case, you can try increasing the latency by increasing `prefillSize` in the Bela code.
