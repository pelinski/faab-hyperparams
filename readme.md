# faab-hyperparams

_this code has been tested with Bela at `dev` commit `71a6e62a`_

Instructions to set up the Bela are in [setup-bela.md](setup-bela.md). If you have been given a microsd card by me, you can skip them. Just open the Bela IDE (type `bela.local`in the browser) and run the `faab-run` project.

# Setup

## 1. Install dependencies

To use `pyaudio` we need `portaudio` installed in our computer. In mac (for other platforms see [pyaudio installation instructions](https://pypi.org/project/PyAudio/)).

```bash
brew install portaudio
```

This project uses `uv` to manage dependencies. _previously we were using pipenv but it's very slow so I switched to `uv`_

```bash
pip install uv
```

Clone this repository with its submodules:

```bash
git clone --recurse-submodules -j8  git@github.com:pelinski/faab-hyperparams.git
```

Install the dependencies with `uv`:

```bash
cd faab-hyperparams # or the folder where you cloned the repo
uv venv
uv pip install torch #==2.0.1+cu117 --index-url https://download.pytorch.org/whl/cu117 # for g15
```

## 2. Download trained models

Download the trained models from [this link](https://www.dropbox.com/scl/fo/3ou91kqehjeb9g30roslq/AM_sA7LBkurbI3OJTwrcx2Y?rlkey=8q9chh9cgiy5nqye6hzh8xdtp&st=onfmepsq&dl=0) and extract them in the `src/models/trained/` folder.

# Running the project

## Run the Bela code

First copy the `faab-run` project to Bela. _this has been modified since last time so you need to do this!_

If the Bela is connected to the laptop you can do, from the project root folder:

```bash
sh scripts/copy.sh bela-code/faab-run
```

Once copied head to the Bela IDE (at `bela.local` in the browser) and run the `faab-run` project. You will need to use the cape to attach the piezo sensors to Bela.

## Run the Python code

The basic command to run the Python code, is, from the project root folder:

```bash
uv run python src/callback.py --audio
```

This will play the output of the model (the sum of the 4 dimensions). There are a few extra options you can add:

```bash
$ uv run python src/callback.py -h
usage: callback.py [-h] [--osc] [--plot] [--out2Bela] [--audio] [--mssd] [--model_type {timelin,timecomp}]

faab-callback

options:
  -h, --help            show this help message and exit
  --osc                 Use OSC server
  --plot                Enable Bokeh plotting
  --out2Bela            Send output to Bela
  --audio               Play model output as audio
  --mssd                Enable MSSD processing
  --model_type {timelin,timecomp}
                        Type of model to use: 'timelin' for TransformerAutoencoder or 'timecomp' for TransformerTimeAutoencoder
```

if you do:

```bash
uv run python src/callback.py --audio --mssd
```

instead of the model output it will play the sonified muliscale spectral difference. You can also add `--plot` to see the signals plots and spectrograms in real-time. Using `--plot` adds some computational load so to avoid audio underruns if uses a longer audio buffer queue which results in a larger latency. Finally, you can add `--timecomp` to use the time compression models. These sound kind of rubbish at the moment but should improve in the next days.

## Extras

If you want to play the piezo signals dataset instead of playing the FAAB, you can run the project `faab-run-nosensors` instead of `faab-run`. You will need to download the [dataset files](https://www.dropbox.com/scl/fo/x4wfj6fknuou7osq69hqq/AO8zZqN4RYf9aiBoH3ZUc_M?rlkey=t2p3czmydct9nbdps8zvlyoku&st=8n16doxf&dl=0) and copy them under `bela-code/faab-nosensors/data/january-2025`.

# Troubleshooting

for the Pepper mode, if you see "Buffer n full" in the Bela console, it means that the data is not being sent quickly enough to Bela. This can happen because you stopped the python code and the Bela code is still running. If this is not the case, you can try increasing the latency by increasing `prefillSize` in the Bela code.
