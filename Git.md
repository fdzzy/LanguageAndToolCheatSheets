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

# Undoing things
https://github.blog/2015-06-08-how-to-undo-almost-anything-with-git/

Ammend
```bash
$ git commit -m 'initial commit'
$ git add forgotten_file
$ git commit --amend
```

Unstaging a Staged File
```bash
$ git reset HEAD <file>
```

Unmodifying a Modified File
```bash
$ git checkout -- <file>
```

Undo the last commit
```bash
# --soft flag: this makes sure that the changes in undone revisions are preserved.
$ git reset --soft HEAD~1
# If you don't want to keep these changes, simply use the --hard flag. Be sure to only do this when you're sure you don't need these changes anymore.
$ git reset --hard HEAD~1
```