# ULLA Ai in Drug Discovery 2023 -- Leiden

Materials for the practicals organized by Leiden University at the 2023 ULLA course AI in DD. Contents:

1. [Multidimensional Data Visualization and Unsupervised Learning](./practicals/01_plotting_unsupervised/)

You will find an `init.sh` script in each folder alongside the provided Jupyter notebooks. This script will install all the required packages and is also executed in the beginning of each notebook. Make sure to check the following if things are not working as expected:

2. Note that after reinitialization of the Notable environment, the `init.sh` scripts need to be executed again for the notebook you want to run. However, all the files you create will be persisted across sessions. 

1. In some cases it might be necessary to restart the notebook's kernel after the `init.sh` runs so it is recommended to do this after each `init.sh` run, especially when ran for the first time in a new Notable environment. Otherwise, you might get import errors for the installed packages even though they are seemingly installed already.

You can download seperate files from this repository with the download options, but it is generally better to clone your own local copy. In order to download this repository in Notable, you can use the following shell command in the Terminal session of Jupyter Lab:

```bash

wget https://raw.githubusercontent.com/CDDLeiden/ULLA2023-Leiden/master/init.sh -O init.sh && chmod +x init.sh && ./init.sh
```

But you can also execute it from a Jupyter Notebook cell (note the `!` in front of the command):

```bash
!wget https://raw.githubusercontent.com/CDDLeiden/ULLA2023-Leiden/master/init.sh -O init.sh && chmod +x init.sh && ./init.sh
```

After you have run these commands, the `ULLA2023-Leiden` folder with the contents of this repository will become available in the working directory. As the course progresses, more tutorials will become available here. You can update your local repository by running the following command in the `ULLA2023-Leiden` folder:

```
git pull origin master
```
