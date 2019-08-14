# System version
Get ubuntu version
```bash
$ lsb_release -a| grep "Release" | awk '{print $2}'
```

Get Linux kernel version
```bash
$ uname -r
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
```

Detach from session:
```
Ctrl a + d
```

Kill a session:
```
Ctrl a + k
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
