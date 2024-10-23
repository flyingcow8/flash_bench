# FlashBench

## Compile the flatbuffers schema
```
flatc --python bench_data.fbs
```
flatc is the linux version, and flatc.exe is the windows version.

## Run the unit tests
```
python -m unittest tests/test_create_solution.py
python -m unittest tests/test_create_bench_table.py
```
## Run the benchmark
```
python bench_main.py
```
