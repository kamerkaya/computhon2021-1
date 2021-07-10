# computhon2021-1: Jaccard Similarity

The graph files
- g0: a very small graph
- dblp is larger, youtube is much larger
- you need to unzip them before using them
- for youtube, single core 10 min execution time may not work - increase it in your script.


## Timings
Using the naive, single-core implementation in the file `jaccard.cpp`, running on an `akya-cuda` server, and compiled with `-O3` optimication, we calculate the Jaccards in the following times:

* `com-dblp`: 22.74 s 
* `youtube`: 3225.27 s
