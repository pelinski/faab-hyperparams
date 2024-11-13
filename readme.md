# faab

this code has been tested with Bela at `dev` commit `71a6e62a`

# 0. Install pipenv

This project uses `pipenv` to manage the dependencies.

```bash
pip install pipenv
```

# 1. Clone this repo on your computer

```bash
git clone --recurse-submodules -j8  git@github.com:pelinski/faab.git
pipenv install
pipenv run pip install torch # --index-url https://download.pytorch.org/whl/cu117 # for g15
```

# 2. Change Bela branch to `dev`

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

# 3. Copy the `faab-run` project to Bela, compile it and run it

With Bela connected to the computer, run the following command (in the computer) to copy the code to Bela:

```bash
sh scripts/copy.sh bela-code/faab-run # to be run from the root folder of the project
```

now you can compile it:

```bash
sh scripts/compile.sh faab-run
```

# 4. Run the python code

Open a new terminal and run the following command:

```bash
pipenv run callback
```
