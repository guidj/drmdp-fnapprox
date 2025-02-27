# DR-MDP


### Box2d setup


Missing `string.h` file:

```
# find path
mdfind -name string.h 
# expose path
export CPATH=FOUND_PATH
# e.g. /Library/Developer/CommandLineTools/SDKs/zyx.sdk/usr/include/c++/v1/
```