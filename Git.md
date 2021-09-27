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

# Fetch upstream
```bash
# make sure you are tracking upstream branch
$ git remote -v
# specify upstream if necessary
$ git remote add upstream https://github.com/ORIGINAL_OWNER/ORIGINAL_REPOSITORY.git
# fetch upstream updates
$ git fetch upstream
# merge upstream changes
$ git checkout master
$ git merge upstream/master
# push to origin
$ git push origin master
```

# Fetch branch on someone else's fork
```bash
$ git remote add theirUsername git@github.com:theirUsername/repoName.git
$ git fetch theirUsername
$ git checkout -b myNameForTheirBranch theirUsername/theirBranch
```

# Stash
```bash
# saving current changes using stash
$ git stash
# check stash list
$ git stash list
# loading saved changes
$ git stash pop

# git stash with a name
$ git stash push -m "my_stash" # my_stash is the stash name
$ git stash list
# to apply a stash and remove it from the stash stack, type:
$ git stash pop stash@{n}
# to apply a stash and keep it in the stash stack, type:
$ git stash apply stash@{n} # where n is the index of the stashed change.
# Notice that you can apply a stash and keep it in the stack by using the stash name
$ git stash apply my_stash_name
# delete a stash
$ git stash drop stash@{n}
```

# submodule
```bash
$ git submodule update

Unmerged paths:
  (use "git restore --staged <file>..." to unstage)
  (use "git add <file>..." to mark resolution)
	both modified:   a_submodule_path
$ git reset HEAD a_submodule_path
```