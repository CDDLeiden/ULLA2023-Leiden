# ULLA Ai in Drug Discovery 2023 -- Leiden

Materials for the practicals organized by Leiden University at the 2023 ULLA course AI in DD. Contents:

1. [Multidimensional Data Visualization and Unsupervised Learning](./practicals/01_plotting_unsupervised/)

You will find an `init.sh` script in each folder alongside the provided Jupyter notebooks. This script will install all the required packages and is also executed in the beginning of each notebook. Note that after reinitialization of the Notable environment, the script needs to be executed again and in some cases it might be necessary to restart the notebook's kernel after the install as well or you might get import errors for the installed packages. However, all the files you create will be persisted across sessions. 

In order to download this repository in Notable, you can use the following shell command in the Terminal session of Jupyter Lab:

```bash

wget https://raw.githubusercontent.com/CDDLeiden/ULLA2023-Leiden/master/init.sh?token=GHSAT0AAAAAABRXDNY4YIB2B2XREGD3IPWYZAI72VQ -O init.sh && chmod +x init.sh && ./init.sh
```

But you can also execute it from a Jupyter Notebook cell (note the `!` in front of the command):

```bash
!wget https://raw.githubusercontent.com/CDDLeiden/ULLA2023-Leiden/master/init.sh?token=GHSAT0AAAAAABRXDNY4YIB2B2XREGD3IPWYZAI72VQ -O init.sh && chmod +x init.sh && ./init.sh
```

After you have run these commands, the `ULLA2023-Leiden` folder with the contents of this repository will become available in the working directory. As the course progresses, more tutorials will become available here. You can update your local repository by running the following command in the `ULLA2023-Leiden` folder:

```
git pull origin master
```
