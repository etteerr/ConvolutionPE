# Existing variables:
#	${bindir} ../bin
#	${common_source} ../src/*c
#	${headers} ../headers
#	${common_libs} -Llib things
#	${common_flags} 


builder = gcc
linker = nvcc

#Flags to pass. Note: These are regardless of debug and release
FLAGS = 
#libs, idem
LIBS = 

#Binary target file
target := ${bindir:%=%/}cuda
source := conv.cu

#--- Common source files and headerfiles from arguments ---
# Adds *.c files from ../src
source += ${common_source}
# Adds libs from common
LIBS += ${common_libs}
# Add flags from common
FLAGS += ${common_flags}
#FLAGS = -std=c11  -fopenmp -march=native -mno-sse4.2 # -mavx

.PHONY: all
all: release

.PHONY: release
#Release settings, adds release flags
release: FLAGS += -g -O3
release: $(target)

.PHONY: debug
#debug, adds debug settings (flags)
debug: FLAGS += -g -O0
debug: $(target)
	
#build all object files (includes common source), headers are also included.
%.c.o : %.c
	$(builder) -c $(FLAGS) $< ${headers:%=-I%} -o $@ $(LIBS)
	
%.cu.o : %.cu
	nvcc -G -c -arch=sm_52 -Xcompiler "$(FLAGS:-std%=)" $< ${headers:%=-I%} -o $@ $(LIBS)
	
$(target) : $(source:%=%.o)
	$(linker) -arch=sm_52 -Xcompiler "$(FLAGS:-std%=)" $(source:%=%.o) -o $@ $(LIBS)
	
clean:
	rm -f *.o
	rm -f $(target)
	rm -f $(target:%=%.asm)
	
asm:
	objdump -S -d $(target) >  $(target:%=%.asm)
	
