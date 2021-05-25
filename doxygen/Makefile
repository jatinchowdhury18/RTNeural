.PHONY: all deploy clean

all:
	doxygen Doxyfile

clean:
	rm -Rf ../docs/**

deploy: all
	scp -r ../docs jatin@ccrma-gate.stanford.edu:~/Library/Web/chowdsp/RTNeural
