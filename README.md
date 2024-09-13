# FlashBench

## Generate the flatbuffers api for python
```
flatc --python bench_data.fbs
```
## Run the tests
```
python -m unittest tests/test_generate_solution.py
python -m unittest tests/test_generate_results_yaml.py
```
## Run the benchmark
```
python bench_main.py
```