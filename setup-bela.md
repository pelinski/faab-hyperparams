# Instructions to set up Bela and run the `faab-run` project

## 1. Change Bela branch to `dev`

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

## 2. Copy the `faab-run` project to Bela, compile it and run it

With Bela connected to the computer, run the following command (in the computer) to copy the code to Bela:

```bash
sh scripts/copy.sh bela-code/faab-run # to be run from the root folder of the project
```

now you can compile it:

```bash
sh scripts/compile.sh faab-run
```
