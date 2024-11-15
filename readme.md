# faab-ai-mami

this code has been tested with Bela at `dev` commit `71a6e62a`

if you have been given a microsd card by me, skip steps 2 and 3, open the Bela IDE (type `bela.local`in the browser) and run the `faab-run` project.

# Setup

## 0. Install pipenv

This project uses `pipenv` to manage the dependencies.

```bash
pip install pipenv
```

## 1. Clone this repo on your computer

```bash
git clone --recurse-submodules -j8  git@github.com:pelinski/faab.git
pipenv install
pipenv run pip install torch # --index-url https://download.pytorch.org/whl/cu117 # for g15
```

## 2. Change Bela branch to `dev`

On your computer

```bash
git clone https://github.com/BelaPlatform/Bela.git
git remote add board root@bela.local:Bela/
git switch -C tmp 71a6e62a
git push -f tmp:tmp
```

ssh into Bela (`ssh root@bela.local`) and change the Bela branch to `tmp`

```bash
# in Bela
cd Bela
git checkout tmp
make -f Makefile.libraries cleanall && make coreclean
```

## 3. Copy the `faab-run` project to Bela, compile it and run it

With Bela connected to the computer, run the following command (in the computer) to copy the code to Bela:

```bash
sh scripts/copy.sh bela-code/faab-run # to be run from the root folder of the project
```

now you can compile it:

```bash
sh scripts/compile.sh faab-run
```

# Running the python code

The Bela code should be running from step 3. If it's not already running, either run the code from step 3 again or run the `faab-run` project from the Bela IDE.

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

for the Pepper mode, if you see "Buffer n full" in the Bela console, it means that the data is not being sent quickly enough to Bela. You can try increasing the latency by increasing `prefillSize` in the Bela code.
