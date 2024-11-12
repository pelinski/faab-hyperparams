# faab

bela at `dev` commit `71a6e62a`

# 1. Clone this repo on your computer

```bash
git clone --recurse-submodules -j8  git@github.com:pelinski/faab.git
pipenv install
pipenv run pip3 install torch # --index-url https://download.pytorch.org/whl/cu117 # for g15
```

# 2. Change Bela branch to `dev`

Follow the instructions [here](https://github.com/BelaPlatform/pybela?tab=readme-ov-file#2-set-the-bela-branch-to-dev). This is necessary for `pybela` to work.

# 3. Copy the `faab-run` project to bela

With Bela connected to the computer, run the following command (in the computer):

```bash
sh scripts/copy.sh bela-code/faab-run # to be run from the root folder of the project
```

# 4. Compile and run the project on Bela

Open a new terminal and run the following command:

```bash
sh scripts/compile.sh faab-run
```

# 5. Run the python code

Open a new terminal and run the following command:

```bash
pipenv run callback
```
