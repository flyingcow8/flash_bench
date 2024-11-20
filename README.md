# FlashBench

## Compile the flatbuffers schema

linux:
```
flatc --python bench_data.fbs
flatc --cpp bench_data.fbs
```
windows:
```
flatc.exe --python bench_data.fbs
flatc.exe --cpp bench_data.fbs
```

## Run the unit tests
```
python -m unittest tests/test_create_solution.py
python -m unittest tests/test_create_bench_table.py
```
## Run the benchmark
```
python bench_main.py
```
