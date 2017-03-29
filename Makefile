test: test.out
	cat test.out > test
test.out: phyloreg/tests/*.py cpp_extensions/*.cpp cpp_extensions/*.h setup.py test.sh
	bash test.sh
