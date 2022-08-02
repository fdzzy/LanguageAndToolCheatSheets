# References
https://devhints.io/bash

# Basics
Variables
```bash
NAME="John"
echo $NAME
echo "$NAME"
echo "${NAME}!"
```

String quotes
```bash
NAME="John"
echo "Hi $NAME"  #=> Hi John
echo 'Hi $NAME'  #=> Hi $NAME
```

Conditions
```bash
git commit && git push
git commit || echo "Commit failed"

if [[ -z "$string" ]]; then
  echo "String is empty"
elif [[ -n "$string" ]]; then
  echo "String is not empty"
fi
```

Loop
```bash
for i in /etc/rc.*; do
  echo $i
done

# reading lines
< file.txt | while read line; do
  echo $line
done

# forever
while true; do
  ···
done

# ranges
for i in {1..5}; do
    echo "Welcome $i"
done
for i in {5..50..5}; do
    echo "Welcome $i"
done

# C-like for loop
for ((i = 0 ; i < 100 ; i++)); do
  echo $i
done
```

# switch
case $COUNTRY in

  Lithuania)
    echo -n "Lithuanian"
    ;;

  Romania | Moldova)
    echo -n "Romanian"
    ;;

  Italy | "San Marino" | Switzerland | "Vatican City")
    echo -n "Italian"
    ;;

  *)
    echo -n "unknown"
    ;;
esac

Functions
```bash
myfunc() {
    echo "hello $1"
}

# Same as above (alternate syntax)
function myfunc() {
    echo "hello $1"
}

myfunc "John"
```

# System version
Get ubuntu version
```bash
$ lsb_release -a| grep "Release" | awk '{print $2}'
```

Get Linux kernel version
```bash
$ uname -r
```

# Zip and unzip
```bash
tar -cvjSf folder_name.tar.bz2 folder_name/

tar -czvf name-of-archive.tar.gz /path/to/directory-or-file
tar -czvf archive.tar.gz /home/ubuntu/Downloads /usr/local/stuff /home/ubuntu/Documents/notes.txt
tar -cjvf archive.tar.bz2 stuff
```

# Find and proces
```bash
find ./ -name "*.txt" -exec grep {} ";"
find ./ -name "*.txt" | xargs grep {} ";"
# upload a bunch of files to aws s3
find ./ -name "*.txt" -exec aws s3 cp {} s3://some_s3_path ";"
```

# Remove certain files on aws s3
```bash
aws s3 ls s3://some_folder | grep some_text | awk '{print $4}' | xargs -I% bash -c 'aws s3 rm s3://some_folder/%'
```

# Rsync
https://stackoverflow.com/questions/12460279/how-to-keep-two-folders-automatically-synchronized
```bash
# sudo apt-get install inotify-tools

while inotifywait -r -e modify,create,delete /directory; do
    rsync -avz /directory /target
done
```

# Screen
Create named screen session
```bash
$ screen -S [name of session]
```

List sessions:
```bash
$ screen -ls
```

Restore session:
```bash
$ screen -r [id of session]
$ screen -d -r [id of session] # force to attach to a session
```

Detach from session:
```
Ctrl a + d
```

Kill a session:
```
Ctrl a + k
or
$ screen -XS <session-id> quit
```

# TMUX
Create named session:
```bash
$ tmux new -s [name of session]
```

List sessions:
```bash
$ tmux ls
```

Restore session:
```bash
$ tmux a -t [name/id of session]
Or
$ tmux attach-session -t [name/id of session]
```

Kill session:
```bash
$ exit (when in session)
```

Detach session:
```bash
Ctrl b + d
```

Split a pane horizontally:
```bash
Ctrl b + "
```

Split a pane vertically:
```bash
Ctrl b + %
```

Move from pane to pane:
```bash
Ctrl b + [arrow key]
```

Kill a window:
```bash
Ctrl b + &
```

Kill a pane:
```bash
Ctrl b + x
```
