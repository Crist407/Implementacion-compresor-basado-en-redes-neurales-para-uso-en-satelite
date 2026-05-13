CC = gcc
MODE ?= release
OMP ?= 1
# Generic flags
LDFLAGS = -lm

CFLAGS_COMMON = -Wall -Wextra -std=c11 -MMD -MP
CFLAGS_DEBUG = -O0 -g
# Release: O3 para velocidad, ftree-vectorize para bucles, fno-fast-math para paridad numérica
CFLAGS_RELEASE = -O3 -DNDEBUG -fno-fast-math -ftree-vectorize
# Release fast: prioriza rendimiento (puede reducir paridad exacta)
CFLAGS_RELEASE_FAST = -O3 -DNDEBUG -ffast-math -fno-math-errno -fno-trapping-math -ftree-vectorize

ifeq ($(MODE),release)
  CFLAGS = $(CFLAGS_COMMON) $(CFLAGS_RELEASE)
else ifeq ($(MODE),release_fast)
  CFLAGS = $(CFLAGS_COMMON) $(CFLAGS_RELEASE_FAST)
else
  CFLAGS = $(CFLAGS_COMMON) $(CFLAGS_DEBUG)
endif

# OpenMP toggle (real): pass OMP=0 para build sin OpenMP
ifeq ($(OMP),1)
  CFLAGS += -fopenmp
  LDFLAGS += -fopenmp
endif

# Auto-detect Architecture
UNAME_M := $(shell uname -m)

ifeq ($(UNAME_M),aarch64)
  # Raspberry Pi 3B+/4 (64-bit)
  CFLAGS += -mcpu=cortex-a53 -mtune=cortex-a53 -DUSE_NEON
  # Note: 64-bit implies NEON support.
else ifeq ($(UNAME_M),armv7l)
  # Raspberry Pi 3 (32-bit Legacy) - Not recommended but supported
  CFLAGS += -mcpu=cortex-a53 -mfpu=neon-fp-armv8 -mfloat-abi=hard -DUSE_NEON
else
  # x86_64 Host (Dev/Test)
  CFLAGS += -march=native -mtune=native 
endif

# Manual Overrides (Optional)
ifeq ($(RPI_ARCH),rpi3)
  CFLAGS += -mcpu=cortex-a53 -mtune=cortex-a53 -DUSE_NEON
endif
ifeq ($(RPI_ARCH),rpi4)
  CFLAGS += -mcpu=cortex-a72 -mtune=cortex-a72 -DUSE_NEON
endif

TARGET_ENC = sorteny_compressor
TARGET_DEC = sorteny_decompressor
TARGET_TEST = sorteny_decoder_ops_test
TARGET_FQ = sorteny_fq_qmap
TARGET_SEM = sorteny_semantic_qmap
SRC_DIR = src/c
INPUT_RAW ?= data/T31TCG_20230907T104629_5.8_512_512_2_1_0.raw
LAMBDA ?= 0.1
MAX_LAMBDA ?= 0.125
ENC_WEIGHTS ?= weights/encoder
DEC_WEIGHTS ?= weights/decoder
LATENT_OUT ?= output/latent.bin
RECON_OUT ?= output/reconstructed.raw
BENCH_THREADS ?= 4

COMMON_SRCS = \
  $(SRC_DIR)/sorteny_model.c \
  $(SRC_DIR)/sorteny_layers.c \
  $(SRC_DIR)/io_helpers.c

ENC_SRCS = $(SRC_DIR)/main.c $(COMMON_SRCS)
DEC_SRCS = $(SRC_DIR)/decompress.c $(COMMON_SRCS)
TEST_SRCS = $(SRC_DIR)/test_decoder_ops.c $(COMMON_SRCS)
FQ_SRCS = $(SRC_DIR)/fixed_quality_qmap.c
SEM_SRCS = $(SRC_DIR)/semantic_qmap.c

ENC_OBJS = $(ENC_SRCS:.c=.o)
DEC_OBJS = $(DEC_SRCS:.c=.o)
TEST_OBJS = $(TEST_SRCS:.c=.o)
FQ_OBJS = $(FQ_SRCS:.c=.o)
SEM_OBJS = $(SEM_SRCS:.c=.o)
DEPS = $(ENC_OBJS:.o=.d) $(DEC_OBJS:.o=.d) $(TEST_OBJS:.o=.d) $(FQ_OBJS:.o=.d) $(SEM_OBJS:.o=.d)

.PHONY: all clean distclean run run_dec run_pipeline run_parity run_fast test_ops rpi3 rpi4 rpi3_fast rpi4_fast

# Permite usar '>' como prefijo de recetas en lugar de tabulador
.RECIPEPREFIX := >

all: $(TARGET_ENC) $(TARGET_DEC) $(TARGET_FQ) $(TARGET_SEM)

$(TARGET_ENC): $(ENC_OBJS)
> @echo Enlazando: $@
> $(CC) $(CFLAGS) -o $@ $(ENC_OBJS) $(LDFLAGS)

$(TARGET_DEC): $(DEC_OBJS)
> @echo Enlazando: $@
> $(CC) $(CFLAGS) -o $@ $(DEC_OBJS) $(LDFLAGS)

$(TARGET_TEST): $(TEST_OBJS)
> @echo Enlazando: $@
> $(CC) $(CFLAGS) -o $@ $(TEST_OBJS) $(LDFLAGS)

$(TARGET_FQ): $(FQ_OBJS)
> @echo Enlazando: $@
> $(CC) $(CFLAGS) -o $@ $(FQ_OBJS) $(LDFLAGS)

$(TARGET_SEM): $(SEM_OBJS)
> @echo Enlazando: $@
> $(CC) $(CFLAGS) -o $@ $(SEM_OBJS) $(LDFLAGS)

%.o: %.c
> @echo Compilando: $<
> $(CC) $(CFLAGS) -c $< -o $@

# Ejecuciones locales con las rutas vigentes del repo.
run: $(TARGET_ENC)
> ./$(TARGET_ENC) $(INPUT_RAW) $(LAMBDA) $(LATENT_OUT) $(ENC_WEIGHTS) $(MAX_LAMBDA)

run_dec: $(TARGET_DEC)
> ./$(TARGET_DEC) $(LATENT_OUT) $(RECON_OUT) $(DEC_WEIGHTS) $(MAX_LAMBDA)

run_pipeline: $(TARGET_ENC) $(TARGET_DEC)
> ./$(TARGET_ENC) $(INPUT_RAW) $(LAMBDA) $(LATENT_OUT) $(ENC_WEIGHTS) $(MAX_LAMBDA)
> ./$(TARGET_DEC) $(LATENT_OUT) $(RECON_OUT) $(DEC_WEIGHTS) $(MAX_LAMBDA)

run_parity: $(TARGET_ENC) $(TARGET_DEC)
> STRICT_PARITY=1 ./$(TARGET_ENC) $(INPUT_RAW) $(LAMBDA) $(LATENT_OUT) $(ENC_WEIGHTS) $(MAX_LAMBDA)
> STRICT_PARITY=1 ./$(TARGET_DEC) $(LATENT_OUT) $(RECON_OUT) $(DEC_WEIGHTS) $(MAX_LAMBDA)

run_fast: $(TARGET_ENC) $(TARGET_DEC)
> OMP_NUM_THREADS=$(BENCH_THREADS) ./$(TARGET_ENC) $(INPUT_RAW) $(LAMBDA) $(LATENT_OUT) $(ENC_WEIGHTS) $(MAX_LAMBDA)
> OMP_NUM_THREADS=$(BENCH_THREADS) ./$(TARGET_DEC) $(LATENT_OUT) $(RECON_OUT) $(DEC_WEIGHTS) $(MAX_LAMBDA)

test_ops: $(TARGET_TEST)
> ./$(TARGET_TEST)

# --- SHORTCUTS PARA RASPBERRY ---

rpi3:
> $(MAKE) MODE=release RPI_ARCH=rpi3 OMP=1

rpi4:
> $(MAKE) MODE=release RPI_ARCH=rpi4 OMP=1

rpi3_fast:
> $(MAKE) MODE=release RPI_ARCH=rpi3 OMP=1 BENCH_THREADS=4

rpi4_fast:
> $(MAKE) MODE=release RPI_ARCH=rpi4 OMP=1 BENCH_THREADS=4

clean:
> @echo Limpiando...
> rm -f $(TARGET_ENC) $(TARGET_DEC) $(TARGET_TEST) $(TARGET_FQ) $(TARGET_SEM) $(ENC_OBJS) $(DEC_OBJS) $(TEST_OBJS) $(FQ_OBJS) $(SEM_OBJS) $(DEPS)

distclean: clean

-include $(DEPS)
