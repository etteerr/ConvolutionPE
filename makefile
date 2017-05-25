#Project targets
SUBDIRS = avx basic optimized cuda

#Common settings
bindir = ${CURDIR}/bin
srcdir = ${CURDIR}/src
common_src = $(wildcard $(srcdir)/*.c)
headers = ${CURDIR}/headers
common_libs = 
common_flags = -fopenmp -march=native -mavx


extra_cmd = 
all: release

.PHONY: subdirs $(SUBDIRS)
.PHONY: all
.PHONY: debug
.PHONY: release
.PHONY: asm


subdirs: $(SUBDIRS)

$(SUBDIRS): directories
	$(MAKE) -C $@ $(extra_cmd) 	bindir="$(bindir)" common_source="$(common_src)" headers="$(headers)" common_libs="$(common_libs)" common_flags="$(common_flags)"
	
debug: extra_cmd = debug
debug: subdirs

release: extra_cmd = release
release: subdirs

asm: extra_cmd = asm
asm: subdirs

.PHONY: directories
directories:
	mkdir -p $(bindir) $(headers) $(srcdir)

test:
	echo $(common_src)
	echo $(common_head)
	echo $(bindir)
	
.PHONY: clean	
clean: extra_cmd = clean
clean: subdirs
	rm -f $(srcdir)/*.o