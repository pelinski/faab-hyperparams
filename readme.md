# faab-ai-mami

this code has been tested with Bela at `dev` commit `71a6e62a`

Instructions to set up the Bela are in [setup-bela.md](setup-bela.md). If you have been given a microsd card by me, you can skip them. Just open the Bela IDE (type `bela.local`in the browser) and run the `faab-run` project.

# Setup

## Install pipenv

This project uses `pipenv` to manage the dependencies.

```bash
pip install pipenv
```

## Clone this repo on your computer

```bash
git clone --recurse-submodules -j8  git@github.com:pelinski/faab.git
```

## Install the python dependencies

```bash
pipenv install
pipenv run pip install torch # --index-url https://download.pytorch.org/whl/cu117 # for g15
```

# Running the python code

If it's not running already, run the `faab-run` project from the Bela IDE.

**Important**: If you stop the python code, you should stop the Bela project as well, otherwise it will give a `Buffer n full` warning. If you want to restart the python code, you should restart the Bela project first.

## for pepper

Open a new terminal and run the following command:

```bash
pipenv run callback
```

## with OSC + SuperCollider

The SuperCollider code to receive the OSC messages is in the `supercollider` folder. Run it in SuperCollider.
To run the python code, open a new terminal and run:

```bash
pipenv run callback-osc
```

# Troubleshooting

for the Pepper mode, if you see "Buffer n full" in the Bela console, it means that the data is not being sent quickly enough to Bela. This can happen because you stopped the python code and the Bela code is still running. If this is not the case, you can try increasing the latency by increasing `prefillSize` in the Bela code.
