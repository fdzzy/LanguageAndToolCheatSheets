# First time git setup

Set username and email
```bash
$ git config --global user.name "John Doe"
$ git config --global user.email johndoe@example.com
```

# Branch
```bash
$ git branch # show branches
$ git checkout existing_branch_name # checkout branch
$ git checkout -b new_branch_name # create and checkout new branch
$ git branch -d branch_name # delete branch
```

Track remote branch
```bash
$ git fetch origin remote_branch_name
$ git checkout -b remote_branch_name --track origin/remote_branch_name
```