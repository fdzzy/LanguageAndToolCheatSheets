# Reference
https://www.csie.ntu.edu.tw/~r92092/ref/win32/win32scripting.html

# Basics
Variables
```bat
set name=value

@echo off
msg1=Hello
msg2=There!
echo %msg1% %msg%
```

Command Line Arguments
```
Command line arguments are treated as special variables within the batch script, the reason I am calling them variables is because they can be changed with the shift command. The command line arguments are enumerated in the following manner %0, %1, %2, %3, %4, %5, %6, %7, %8 and %9. %0 is special in that it corresponds to the name of the batch file itself. %1 is the first argument, %2 is the second argument and so on. To reference after the ninth argument you must use the shift command to shift the arguments 1 variable to the left so that $2 becomes $1, $1 becomes $0 and so on, $0 gets scrapped because it has nowhere to go, this can be useful to process all the arguments using a loop, using one variable to reference the first argument and shifting until you have exhausted the arguments list.
```

Conditions
```bat
IF [NOT] ERRORLEVEL number command
IF [NOT] string1==string2 command
IF [NOT] EXIST filename command


@echo off
if "%1"=="1" echo The first choice is nice
if "%1"=="2" echo The second choice is just as nice 
if "%1"=="3" echo The third choice is excellent 
if "%1"==""  echo I see you were wise enough not to choose, You Win!
```

Loop
```bat
FOR %%variable IN (set) DO command [command-parameters]

@echo off
if exist bigtxt.txt rename bigtxt.txt bigtxt
for %%f in (*.txt) do type %%f >> bigtxt
rename bigtxt bigtxt.txt
```