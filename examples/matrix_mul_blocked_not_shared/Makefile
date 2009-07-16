###  Pathes

## Generic
PPU_CC            = ppu32-g++
PPU_STRIP         = ppu-strip
PPU_EMBEDSPU      = ppu32-embedspu
SPU_CC            = spu-gcc
#SPU_CC            = $(SPU_COMPILER_PATH)spuxlc
SPU_STRIP         = spu-strip


PPU_COMPILER_PATH = /opt/ibmcmp/xlc/cbe/10.1/bin/
SPU_COMPILER_PATH = /opt/ibmcmp/xlc/cbe/10.1/bin/
INCLUDE_PATH      = -I../../include/
PPU_INCLUDE_PATH  = -I/home/jens/boost_1_38_0 -I/opt/cell/sdk/usr/include
SPU_INCLUDE_PATH  = -I/opt/cell/sdk/usr/spu/include
LIBRARY_PATH      = 
PPU_LIBARY_PATH   = -L/opt/cell/sdk/usr/lib
SPU_LIBARY_PATH   = -L/opt/cell/sdk/usr/spu/lib


### Compiler FLAGS

## Generic
#PROFILE           = -pg
DEBUG             = -g

## PPU
PPU_WARNING       = -W -Wall
PPU_OPTIMIZE      = -O3
PPU_CFLAGS        = $(PPU_WARNING) $(DEBUG) ${PPU_OPTIMIZE} $(PROFILE)

## SPU
#SPU_WARNING       = -Wno-main
SPU_OPTIMIZE      = -funroll-loops -ftree-vectorize -O3
#SPU_CACHE_FLAGS   = -qswcache_size=128 -qea32
SPU_CFLAGS        = $(DEBUG) $(WARNING) ${SPU_WARNING} ${SPU_OPTIMIZE} ${SPU_CACHE_FLAGS} $(PROFILE)


### Program details

## Generic
LIBARIES          = 

## PPU
PPU_SOURCES       = main.cpp
PPU_OBJECTS       = ${PPU_SOURCES:%.cpp=%.o}
PPU_INCLUDE_PATH  =
PPU_LIBRARIES     = -lspe2 -lpthread
PPU_PROGRAM       = main

## SPU
SPU_SOURCES       = kernel.c
SPU_OBJECTS       = ${SPU_SOURCES:%.c=%.out}
SPU_OBJECTS_EMB   = $(SPU_OBJECTS:%.out=%_em.a)
SPU_SYMBOL_EMB    = $(SPU_SOURCES:%.c=%)
SPU_LIBRARIES     = -lsync -lm


### Targets

## "Normal" targets
all:		clean spu strip-spu embed-spu ppu
		@$(MAKE) $(PPU_PROGRAM)

ppu: 		spu strip-spu embed-spu ${PPU_OBJECTS}
	        

spu:		${SPU_OBJECTS}

embed-spu:	spu strip-spu
		@echo -n "Create SPU module ... "; \
		declare symbol_spu_emb=($(SPU_SYMBOL_EMB)); \
		declare objects_spu_emb=($(SPU_OBJECTS_EMB)); \
		declare objects_spu=($(SPU_OBJECTS)); \
		element_count=$${#symbol_spu_emb[*]} ;\
		for ((i = 0; i < $$element_count; ++i)); \
		do \
			$(PPU_EMBEDSPU) $${symbol_spu_emb[$$i]} $${objects_spu[$$i]} $${objects_spu_emb[$$i]}; \
		done; \
		echo Done.

strip-spu:	spu
#		@echo -n "Do strip SPU objects ... "; \
#		$(SPU_STRIP) -s $(SPU_OBJECTS); \
#		echo "Done."

strip-ppu:	ppu
		@echo -n "Strip PPU objects ... "; \
		$(PPU_STRIP) -s $(PPU_OBJECTS) $(APPNAME); \
		echo "Done."

strip-all:	strip-spu strip-ppu

clean:
		@echo -n "Make clean in all Subdirectories ... "; \
		rm -f $(PPU_OBJECTS) $(SPU_OBJECTS) $(SPU_OBJECTS_EMB) $(PPU_OBJECTS); \
		echo Done.


## Object targets
${SPU_OBJECTS}:
		$(SPU_CC) $(SPU_CFLAGS) $(INCLUDE_PATH) $(SPU_INCLUDE_PATH) $(LIBRARY_PATH) $(SPU_LIBARY_PATH) $(LIBARIES) -o $*.out $*.c $(SPU_LIBRARIES)

${PPU_OBJECTS}:
		$(PPU_CC) -c $(PPU_CFLAGS) $(INCLUDE_PATH) $(PPU_INCLUDE_PATH) $(LIBRARY_PATH) $(PPU_LIBARY_PATH) -o $*.o $*.cpp $(PPU_LIBARIES)

$(PPU_PROGRAM):	$(SPU_OBJECTS) $(PPU_OBJECTS)
		$(PPU_CC) $(PPU_CFLAGS) $(INCLUDE_PATH) $(PPU_INCLUDE_PATH) $(LIBRARY_PATH) $(PPU_LIBARY_PATH) $(LIBARIES) -o $(PPU_PROGRAM) $(PPU_OBJECTS) $(SPU_OBJECTS_EMB) $(PPU_LIBRARIES)

